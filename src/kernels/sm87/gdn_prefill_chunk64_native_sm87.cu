#include "gdn_prefill_chunk64_native_sm87.h"

#include "q3x/runtime/gdn_decode.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace q3x::runtime::gdn_prefill_chunk64_native_detail {
namespace {

namespace wmma = nvcuda::wmma;
using Bf16 = __nv_bfloat16;
using Accumulator = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

constexpr unsigned int kThreads = 256U;
constexpr unsigned int kWarps = 8U;
constexpr unsigned int kWarpSize = 32U;
constexpr unsigned int kTile = 16U;
constexpr unsigned int kTokenCount = 512U;
constexpr unsigned int kChunk = 64U;
constexpr unsigned int kChunks = kTokenCount / kChunk;
constexpr unsigned int kDimension = kGdnHeadDimension;
constexpr unsigned int kValueHeads = kGdnValueHeadCount;
constexpr unsigned int kQkHeads = kGdnQkHeadCount;
constexpr unsigned int kDimensionTiles = kDimension / kTile;
constexpr unsigned int kChunkTiles = kChunk / kTile;
constexpr unsigned int kFullMask = 0xffffffffU;
constexpr std::size_t kQOffset = 0U;
constexpr std::size_t kKOffset = kGdnQElements;
constexpr std::size_t kVOffset = kGdnQElements + kGdnKElements;

// Phase-one shared layout. Later phases recycle the first 32 KiB as
// transform/QK/scratch. This is below the SM87 per-block opt-in ceiling and
// intentionally admits one persistent CTA per SM.
constexpr std::size_t kStateBf16Offset = 0U;
constexpr std::size_t kStateBf16Bytes =
    kDimension * kDimension * sizeof(Bf16);
constexpr std::size_t kQOffsetBytes = kStateBf16Bytes;
constexpr std::size_t kHeadTokenBytes =
    kChunk * kDimension * sizeof(Bf16);
constexpr std::size_t kKOffsetBytes = kQOffsetBytes + kHeadTokenBytes;
constexpr std::size_t kVOffsetBytes = kKOffsetBytes + kHeadTokenBytes;
constexpr std::size_t kStateQOffsetBytes = kVOffsetBytes + kHeadTokenBytes;
constexpr std::size_t kStateQBytes =
    kDimension * kChunk * sizeof(float);
constexpr std::size_t kVcorrOffsetBytes = kStateQOffsetBytes + kStateQBytes;
constexpr std::size_t kVcorrBytes = kChunk * kDimension * sizeof(Bf16);
constexpr std::size_t kScratchOffsetBytes = kVcorrOffsetBytes + kVcorrBytes;
constexpr std::size_t kScratchBytes =
    kWarps * kTile * kTile * sizeof(float);
constexpr std::size_t kGammaOffsetBytes = kScratchOffsetBytes + kScratchBytes;
constexpr std::size_t kGammaBytes = kChunk * sizeof(float);
constexpr std::size_t kBetaOffsetBytes = kGammaOffsetBytes + kGammaBytes;
constexpr std::size_t kSharedBytes = kBetaOffsetBytes + kGammaBytes;

static_assert(kSharedBytes == 139776U);
static_assert(kQkHeads * 3U == kValueHeads);
static_assert(kDimensionTiles == kWarps);

__device__ __forceinline__ float decode_bf16(const std::uint16_t value) {
  return __uint_as_float(static_cast<unsigned int>(value) << 16U);
}

__device__ __forceinline__ std::uint16_t encode_bf16(const float value) {
  const unsigned int bits = __float_as_uint(value);
  if ((bits & 0x7fffffffU) > 0x7f800000U) {
    return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
  }
  return __bfloat16_as_ushort(__float2bfloat16_rn(value));
}

__device__ __forceinline__ float softplus(const float value) {
  return value > 20.0F ? value : log1pf(expf(value));
}

__device__ __forceinline__ float sigmoid(const float value) {
  if (value >= 0.0F) {
    return 1.0F / (1.0F + expf(-value));
  }
  const float exponential = expf(value);
  return exponential / (1.0F + exponential);
}

__device__ __forceinline__ float* warp_scratch(
    unsigned char* const shared, const unsigned int warp) {
  return reinterpret_cast<float*>(shared + kScratchOffsetBytes) +
         warp * kTile * kTile;
}

// Native C64/WY cell. No scalar token recurrence is present: all four dense
// chunk products (State@Q/K, A@Vcorr, NewV@QK, and NewV^T@Kdecay) execute on
// BF16 Tensor Cores. Only the 64x64 causal triangular solve is scalar; it is
// a chunk-level dependency and remains resident in shared memory.
__launch_bounds__(kThreads, 1) __global__ void gdn_c512_wy_kernel(
    const std::uint16_t* const conv_qkv,
    const std::uint16_t* const a,
    const std::uint16_t* const b,
    const std::uint16_t* const A_log,
    const std::uint16_t* const dt_bias,
    const std::uint16_t* const state_input,
    std::uint16_t* const state_output,
    const float l2_epsilon,
    std::uint16_t* const output) {
  extern __shared__ unsigned char shared[];
  const unsigned int thread = threadIdx.x;
  const unsigned int warp = thread / kWarpSize;
  const unsigned int lane = thread % kWarpSize;
  const unsigned int value_head = blockIdx.x;
  const unsigned int qk_head = value_head / 3U;
  const std::size_t head_state =
      static_cast<std::size_t>(value_head) * kDimension * kDimension;

  // Load the initial BF16 state through FP32 shared storage into persistent
  // accumulator fragments. Each warp owns one 16-row state slab.
  auto* state_load = reinterpret_cast<float*>(shared);
  for (unsigned int index = thread; index < kDimension * kDimension;
       index += kThreads) {
    state_load[index] = decode_bf16(state_input[head_state + index]);
  }
  __syncthreads();
  Accumulator state[kDimensionTiles];
#pragma unroll
  for (unsigned int key_tile = 0U; key_tile < kDimensionTiles; ++key_tile) {
    wmma::load_matrix_sync(
        state[key_tile],
        state_load + warp * kTile * kDimension + key_tile * kTile,
        static_cast<int>(kDimension), wmma::mem_row_major);
  }
  __syncthreads();

#pragma unroll 1
  for (unsigned int chunk_index = 0U; chunk_index < kChunks;
       ++chunk_index) {
    auto* const state_bf16 =
        reinterpret_cast<Bf16*>(shared + kStateBf16Offset);
    auto* const shared_q = reinterpret_cast<Bf16*>(shared + kQOffsetBytes);
    auto* const shared_k = reinterpret_cast<Bf16*>(shared + kKOffsetBytes);
    auto* const shared_v = reinterpret_cast<Bf16*>(shared + kVOffsetBytes);
    auto* const state_q =
        reinterpret_cast<float*>(shared + kStateQOffsetBytes);
    auto* const v_corr =
        reinterpret_cast<Bf16*>(shared + kVcorrOffsetBytes);
    auto* const gamma = reinterpret_cast<float*>(shared + kGammaOffsetBytes);
    auto* const beta = reinterpret_cast<float*>(shared + kBetaOffsetBytes);

    // Publish exactly one BF16 state boundary for the chunk's State@Q/K
    // products while keeping the authoritative FP32 fragments live.
#pragma unroll
    for (unsigned int key_tile = 0U; key_tile < kDimensionTiles; ++key_tile) {
      float* const tile = warp_scratch(shared, warp);
      wmma::store_matrix_sync(tile, state[key_tile], static_cast<int>(kTile),
                              wmma::mem_row_major);
      __syncwarp();
      for (unsigned int element = lane; element < kTile * kTile;
           element += kWarpSize) {
        const unsigned int row = element / kTile;
        const unsigned int column = element % kTile;
        state_bf16[(warp * kTile + row) * kDimension +
                   key_tile * kTile + column] =
            __float2bfloat16_rn(tile[element]);
      }
      __syncwarp();
    }
    __syncthreads();

    // Eight warps normalize eight Q/K rows at a time. Q/K are rounded once
    // to the same BF16 boundary used by the screened C64 reference.
#pragma unroll
    for (unsigned int wave = 0U; wave < kChunk / kWarps; ++wave) {
      const unsigned int token_in_chunk = wave * kWarps + warp;
      const unsigned int token = chunk_index * kChunk + token_in_chunk;
      const std::size_t token_base =
          static_cast<std::size_t>(token) * kGdnQkvChannels;
      float q_values[4];
      float k_values[4];
      float q_sum = 0.0F;
      float k_sum = 0.0F;
#pragma unroll
      for (unsigned int item = 0U; item < 4U; ++item) {
        const unsigned int dimension = lane + item * kWarpSize;
        q_values[item] = decode_bf16(
            conv_qkv[token_base + kQOffset + qk_head * kDimension +
                     dimension]);
        k_values[item] = decode_bf16(
            conv_qkv[token_base + kKOffset + qk_head * kDimension +
                     dimension]);
        q_sum = fmaf(q_values[item], q_values[item], q_sum);
        k_sum = fmaf(k_values[item], k_values[item], k_sum);
      }
#pragma unroll
      for (unsigned int offset = 16U; offset != 0U; offset >>= 1U) {
        q_sum += __shfl_down_sync(kFullMask, q_sum, offset);
        k_sum += __shfl_down_sync(kFullMask, k_sum, offset);
      }
      const float q_scale = __shfl_sync(
          kFullMask,
          lane == 0U ? rsqrtf(q_sum + l2_epsilon) *
                           rsqrtf(static_cast<float>(kDimension))
                     : 0.0F,
          0U);
      const float k_scale = __shfl_sync(
          kFullMask,
          lane == 0U ? rsqrtf(k_sum + l2_epsilon) : 0.0F, 0U);
#pragma unroll
      for (unsigned int item = 0U; item < 4U; ++item) {
        const unsigned int dimension = lane + item * kWarpSize;
        shared_q[token_in_chunk * kDimension + dimension] =
            __float2bfloat16_rn(q_values[item] * q_scale);
        shared_k[token_in_chunk * kDimension + dimension] =
            __float2bfloat16_rn(k_values[item] * k_scale);
      }
      if (lane == 0U) {
        const std::size_t scalar =
            static_cast<std::size_t>(token) * kValueHeads + value_head;
        const float gate_input = decode_bf16(a[scalar]) +
                                 decode_bf16(dt_bias[value_head]);
        gamma[token_in_chunk] =
            -expf(decode_bf16(A_log[value_head])) * softplus(gate_input);
        beta[token_in_chunk] = sigmoid(decode_bf16(b[scalar]));
      }
    }
    for (unsigned int index = thread; index < kChunk * kDimension;
         index += kThreads) {
      const unsigned int token_in_chunk = index / kDimension;
      const unsigned int value_dimension = index % kDimension;
      const unsigned int token = chunk_index * kChunk + token_in_chunk;
      const std::size_t source =
          static_cast<std::size_t>(token) * kGdnQkvChannels + kVOffset +
          value_head * kDimension + value_dimension;
      reinterpret_cast<std::uint16_t*>(shared_v)[index] = conv_qkv[source];
    }
    __syncthreads();
    if (thread == 0U) {
      float cumulative = 0.0F;
#pragma unroll
      for (unsigned int token = 0U; token < kChunk; ++token) {
        cumulative += gamma[token];
        gamma[token] = cumulative;
      }
    }
    __syncthreads();

    // State@Q and State@K. State@Q stays FP32. State@K is immediately
    // decayed and transposed into Vcorr, so no W matrix is materialized.
#pragma unroll
    for (unsigned int token_tile = 0U; token_tile < kChunkTiles;
         ++token_tile) {
      Accumulator sq;
      Accumulator sk;
      wmma::fill_fragment(sq, 0.0F);
      wmma::fill_fragment(sk, 0.0F);
#pragma unroll
      for (unsigned int key_tile = 0U; key_tile < kDimensionTiles;
           ++key_tile) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, Bf16,
                       wmma::row_major>
            state_fragment;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, Bf16,
                       wmma::col_major>
            q_fragment;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, Bf16,
                       wmma::col_major>
            k_fragment;
        wmma::load_matrix_sync(
            state_fragment,
            state_bf16 + warp * kTile * kDimension + key_tile * kTile,
            static_cast<int>(kDimension));
        wmma::load_matrix_sync(
            q_fragment,
            shared_q + token_tile * kTile * kDimension + key_tile * kTile,
            static_cast<int>(kDimension));
        wmma::load_matrix_sync(
            k_fragment,
            shared_k + token_tile * kTile * kDimension + key_tile * kTile,
            static_cast<int>(kDimension));
        wmma::mma_sync(sq, state_fragment, q_fragment, sq);
        wmma::mma_sync(sk, state_fragment, k_fragment, sk);
      }
      wmma::store_matrix_sync(
          state_q + warp * kTile * kChunk + token_tile * kTile, sq,
          static_cast<int>(kChunk), wmma::mem_row_major);
      float* const tile = warp_scratch(shared, warp);
      wmma::store_matrix_sync(tile, sk, static_cast<int>(kTile),
                              wmma::mem_row_major);
      __syncwarp();
      for (unsigned int element = lane; element < kTile * kTile;
           element += kWarpSize) {
        const unsigned int row = element / kTile;
        const unsigned int column = element % kTile;
        const unsigned int token_in_chunk = token_tile * kTile + column;
        const unsigned int value_dimension = warp * kTile + row;
        const float prediction =
            tile[element] * expf(gamma[token_in_chunk]);
        const float value = __bfloat162float(
            shared_v[token_in_chunk * kDimension + value_dimension]);
        v_corr[token_in_chunk * kDimension + value_dimension] =
            __float2bfloat16_rn(value - prediction);
      }
      __syncwarp();
    }
    __syncthreads();

    // The old state boundary is dead. Reuse its 32 KiB for FP32 L/inverse,
    // then compact to BF16 A=inv(L)diag(beta) and QK.
    auto* const lower = reinterpret_cast<float*>(shared);
    auto* const inverse =
        reinterpret_cast<float*>(shared + kChunk * kChunk * sizeof(float));
    for (unsigned int index = thread; index < kChunk * kChunk;
         index += kThreads) {
      const unsigned int row = index / kChunk;
      const unsigned int column = index % kChunk;
      float value = 0.0F;
      if (row == column) {
        value = 1.0F;
      } else if (row > column) {
        float dot = 0.0F;
#pragma unroll
        for (unsigned int dimension = 0U; dimension < kDimension;
             ++dimension) {
          dot = fmaf(__bfloat162float(
                         shared_k[row * kDimension + dimension]),
                     __bfloat162float(
                         shared_k[column * kDimension + dimension]),
                     dot);
        }
        value = beta[row] * expf(gamma[row] - gamma[column]) * dot;
      }
      lower[index] = value;
    }
    __syncthreads();
    if (thread < kChunk) {
      const unsigned int column = thread;
#pragma unroll 1
      for (unsigned int row = 0U; row < kChunk; ++row) {
        float value = 0.0F;
        if (row == column) {
          value = beta[column];
        } else if (row > column) {
          float sum = 0.0F;
#pragma unroll 1
          for (unsigned int inner = column; inner < row; ++inner) {
            sum = fmaf(lower[row * kChunk + inner],
                       inverse[inner * kChunk + column], sum);
          }
          value = -sum;
        }
        inverse[row * kChunk + column] = value;
      }
    }
    __syncthreads();
    auto* const transform = reinterpret_cast<Bf16*>(shared);
    auto* const qk = reinterpret_cast<Bf16*>(
        shared + kChunk * kChunk * sizeof(Bf16));
    for (unsigned int index = thread; index < kChunk * kChunk;
         index += kThreads) {
      transform[index] = __float2bfloat16_rn(inverse[index]);
    }
    __syncthreads();

    // NewV=A@Vcorr. V is dead, so its 16 KiB bank becomes NewV.
    auto* const new_v = shared_v;
    for (unsigned int tile_index = warp;
         tile_index < kChunkTiles * kDimensionTiles;
         tile_index += kWarps) {
      const unsigned int token_tile = tile_index / kDimensionTiles;
      const unsigned int value_tile = tile_index % kDimensionTiles;
      Accumulator accumulator;
      wmma::fill_fragment(accumulator, 0.0F);
#pragma unroll
      for (unsigned int source_tile = 0U; source_tile < kChunkTiles;
           ++source_tile) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, Bf16,
                       wmma::row_major>
            a_fragment;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, Bf16,
                       wmma::row_major>
            v_fragment;
        wmma::load_matrix_sync(
            a_fragment,
            transform + token_tile * kTile * kChunk + source_tile * kTile,
            static_cast<int>(kChunk));
        wmma::load_matrix_sync(
            v_fragment,
            v_corr + source_tile * kTile * kDimension + value_tile * kTile,
            static_cast<int>(kDimension));
        wmma::mma_sync(accumulator, a_fragment, v_fragment, accumulator);
      }
      float* const tile = warp_scratch(shared, warp);
      wmma::store_matrix_sync(tile, accumulator, static_cast<int>(kTile),
                              wmma::mem_row_major);
      __syncwarp();
      for (unsigned int element = lane; element < kTile * kTile;
           element += kWarpSize) {
        const unsigned int row = element / kTile;
        const unsigned int column = element % kTile;
        new_v[(token_tile * kTile + row) * kDimension +
              value_tile * kTile + column] =
            __float2bfloat16_rn(tile[element]);
      }
      __syncwarp();
    }
    __syncthreads();

    // QK[source,query], including causal decay, remains shared.
    for (unsigned int tile_index = warp;
         tile_index < kChunkTiles * kChunkTiles; tile_index += kWarps) {
      const unsigned int source_tile = tile_index / kChunkTiles;
      const unsigned int query_tile = tile_index % kChunkTiles;
      Accumulator accumulator;
      wmma::fill_fragment(accumulator, 0.0F);
#pragma unroll
      for (unsigned int key_tile = 0U; key_tile < kDimensionTiles;
           ++key_tile) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, Bf16,
                       wmma::row_major>
            k_fragment;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, Bf16,
                       wmma::col_major>
            q_fragment;
        wmma::load_matrix_sync(
            k_fragment,
            shared_k + source_tile * kTile * kDimension + key_tile * kTile,
            static_cast<int>(kDimension));
        wmma::load_matrix_sync(
            q_fragment,
            shared_q + query_tile * kTile * kDimension + key_tile * kTile,
            static_cast<int>(kDimension));
        wmma::mma_sync(accumulator, k_fragment, q_fragment, accumulator);
      }
      float* const tile = warp_scratch(shared, warp);
      wmma::store_matrix_sync(tile, accumulator, static_cast<int>(kTile),
                              wmma::mem_row_major);
      __syncwarp();
      for (unsigned int element = lane; element < kTile * kTile;
           element += kWarpSize) {
        const unsigned int row = element / kTile;
        const unsigned int column = element % kTile;
        const unsigned int source = source_tile * kTile + row;
        const unsigned int query = query_tile * kTile + column;
        const float score = query >= source
                                ? tile[element] *
                                      expf(gamma[query] - gamma[source])
                                : 0.0F;
        qk[source * kChunk + query] = __float2bfloat16_rn(score);
      }
      __syncwarp();
    }
    __syncthreads();

    for (unsigned int index = thread; index < kDimension * kChunk;
         index += kThreads) {
      state_q[index] *= expf(gamma[index % kChunk]);
    }
    __syncthreads();

    // O=exp(gamma)StateQ + NewV^T@QK, entirely FP32 until the established
    // raw-output BF16 boundary.
#pragma unroll
    for (unsigned int query_tile = 0U; query_tile < kChunkTiles;
         ++query_tile) {
      Accumulator accumulator;
      wmma::load_matrix_sync(
          accumulator,
          state_q + warp * kTile * kChunk + query_tile * kTile,
          static_cast<int>(kChunk), wmma::mem_row_major);
#pragma unroll
      for (unsigned int source_tile = 0U; source_tile < kChunkTiles;
           ++source_tile) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, Bf16,
                       wmma::col_major>
            value_fragment;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, Bf16,
                       wmma::row_major>
            score_fragment;
        wmma::load_matrix_sync(
            value_fragment,
            new_v + source_tile * kTile * kDimension + warp * kTile,
            static_cast<int>(kDimension));
        wmma::load_matrix_sync(
            score_fragment,
            qk + source_tile * kTile * kChunk + query_tile * kTile,
            static_cast<int>(kChunk));
        wmma::mma_sync(accumulator, value_fragment, score_fragment,
                       accumulator);
      }
      wmma::store_matrix_sync(
          state_q + warp * kTile * kChunk + query_tile * kTile, accumulator,
          static_cast<int>(kChunk), wmma::mem_row_major);
    }
    __syncthreads();

    // State_end=decay*State + NewV^T@Kdecay. K is dead after QK and is
    // transformed in place to its end-decayed BF16 form.
    const float final_gamma = gamma[kChunk - 1U];
    for (unsigned int index = thread; index < kChunk * kDimension;
         index += kThreads) {
      const unsigned int token = index / kDimension;
      shared_k[index] = __float2bfloat16_rn(
          __bfloat162float(shared_k[index]) *
          expf(final_gamma - gamma[token]));
    }
    __syncthreads();
    const float state_decay = expf(final_gamma);
#pragma unroll
    for (unsigned int key_tile = 0U; key_tile < kDimensionTiles; ++key_tile) {
#pragma unroll
      for (unsigned int element = 0U; element < state[key_tile].num_elements;
           ++element) {
        state[key_tile].x[element] *= state_decay;
      }
#pragma unroll
      for (unsigned int source_tile = 0U; source_tile < kChunkTiles;
           ++source_tile) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, Bf16,
                       wmma::col_major>
            value_fragment;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, Bf16,
                       wmma::row_major>
            key_fragment;
        wmma::load_matrix_sync(
            value_fragment,
            new_v + source_tile * kTile * kDimension + warp * kTile,
            static_cast<int>(kDimension));
        wmma::load_matrix_sync(
            key_fragment,
            shared_k + source_tile * kTile * kDimension + key_tile * kTile,
            static_cast<int>(kDimension));
        wmma::mma_sync(state[key_tile], value_fragment, key_fragment,
                       state[key_tile]);
      }
    }

    for (unsigned int index = thread; index < kDimension * kChunk;
         index += kThreads) {
      const unsigned int value_dimension = index / kChunk;
      const unsigned int token_in_chunk = index % kChunk;
      const std::size_t destination =
          static_cast<std::size_t>(chunk_index * kChunk + token_in_chunk) *
              kGdnVElements +
          value_head * kDimension + value_dimension;
      output[destination] = encode_bf16(state_q[index]);
    }
    __syncthreads();
  }

  // Persist only the final BF16 state. Intermediate chunk boundaries never
  // leave the CTA.
#pragma unroll
  for (unsigned int key_tile = 0U; key_tile < kDimensionTiles; ++key_tile) {
    float* const tile = warp_scratch(shared, warp);
    wmma::store_matrix_sync(tile, state[key_tile], static_cast<int>(kTile),
                            wmma::mem_row_major);
    __syncwarp();
    for (unsigned int element = lane; element < kTile * kTile;
         element += kWarpSize) {
      const unsigned int row = element / kTile;
      const unsigned int column = element % kTile;
      const std::size_t destination =
          head_state + (warp * kTile + row) * kDimension +
          key_tile * kTile + column;
      state_output[destination] = encode_bf16(tile[element]);
    }
    __syncwarp();
  }
}

__launch_bounds__(128) __global__ void rms_norm_silu_gate_kernel(
    const std::uint16_t* const norm_weight,
    const std::uint16_t* const silu_gate,
    const float epsilon,
    std::uint16_t* const output) {
  __shared__ float squares[kDimension];
  const unsigned int dimension = threadIdx.x;
  const std::size_t token_head = blockIdx.x;
  const std::size_t base = token_head * kDimension;
  const float raw = decode_bf16(output[base + dimension]);
  squares[dimension] = raw * raw;
  __syncthreads();
#pragma unroll
  for (unsigned int stride = kDimension / 2U; stride != 0U; stride >>= 1U) {
    if (dimension < stride) {
      squares[dimension] += squares[dimension + stride];
    }
    __syncthreads();
  }
  const float inverse_rms =
      rsqrtf(squares[0] / static_cast<float>(kDimension) + epsilon);
  const float gate = decode_bf16(silu_gate[base + dimension]);
  const float normalized =
      raw * inverse_rms * decode_bf16(norm_weight[dimension]);
  output[base + dimension] =
      encode_bf16(normalized * gate / (1.0F + expf(-gate)));
}

[[nodiscard]] bool invalid_arguments(
    const std::uint16_t* const conv_qkv,
    const std::size_t token_count,
    const std::uint16_t* const a,
    const std::uint16_t* const b,
    const std::uint16_t* const A_log,
    const std::uint16_t* const dt_bias,
    const std::uint16_t* const state_input,
    const std::uint16_t* const state_output,
    const float l2_epsilon,
    const std::uint16_t* const norm_weight,
    const std::uint16_t* const silu_gate,
    const float norm_epsilon,
    const std::uint16_t* const output) noexcept {
  return token_count != kTokenCount || conv_qkv == nullptr || a == nullptr ||
         b == nullptr || A_log == nullptr || dt_bias == nullptr ||
         state_input == nullptr || state_output == nullptr ||
         norm_weight == nullptr || silu_gate == nullptr || output == nullptr ||
         !std::isfinite(l2_epsilon) || l2_epsilon <= 0.0F ||
         !std::isfinite(norm_epsilon) || norm_epsilon <= 0.0F;
}

}  // namespace

int launch(const std::uint16_t* const conv_qkv,
           const std::size_t token_count,
           const std::uint16_t* const a,
           const std::uint16_t* const b,
           const std::uint16_t* const A_log,
           const std::uint16_t* const dt_bias,
           const std::uint16_t* const state_input,
           std::uint16_t* const state_output,
           const float l2_epsilon,
           const std::uint16_t* const norm_weight,
           const std::uint16_t* const silu_gate,
           const float norm_epsilon,
           std::uint16_t* const output,
           void* const cuda_stream) noexcept {
  if (invalid_arguments(conv_qkv, token_count, a, b, A_log, dt_bias,
                        state_input, state_output, l2_epsilon, norm_weight,
                        silu_gate, norm_epsilon, output)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  cudaError_t status = cudaFuncSetAttribute(
      gdn_c512_wy_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(kSharedBytes));
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  gdn_c512_wy_kernel<<<kValueHeads, kThreads, kSharedBytes, stream>>>(
      conv_qkv, a, b, A_log, dt_bias, state_input, state_output, l2_epsilon,
      output);
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  rms_norm_silu_gate_kernel<<<kTokenCount * kValueHeads, kDimension, 0U,
                              stream>>>(norm_weight, silu_gate, norm_epsilon,
                                       output);
  return static_cast<int>(cudaGetLastError());
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
  cudaError_t status = cudaFuncSetAttribute(
      gdn_c512_wy_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(kSharedBytes));
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  cudaFuncAttributes attributes{};
  status = cudaFuncGetAttributes(&attributes, gdn_c512_wy_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active, gdn_c512_wy_kernel, static_cast<int>(kThreads), kSharedBytes);
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

}  // namespace q3x::runtime::gdn_prefill_chunk64_native_detail
