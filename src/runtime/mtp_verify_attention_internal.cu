/*
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// Internal admission wrapper. v7 preserves scalar arithmetic and pipelines KV
// movement; the previous FlashInfer numerical lineage remains in Git history.
#include "decode_fused_gqa_internal.h"
#include "mtp_verify_graph_internal.h"
#include "q3x/runtime/decode_ops.h"
#include <cuda_runtime.h>
#include <array>
#include <limits>

namespace q3x::runtime::mtp_detail {
namespace {
__device__ __forceinline__ float decode_bf16_device(
    const std::uint16_t value) {
  return __uint_as_float(static_cast<unsigned int>(value) << 16U);
}

__device__ __forceinline__ std::uint16_t encode_bf16_device(
    const float value) {
  unsigned int bits = __float_as_uint(value);
  const unsigned int magnitude = bits & 0x7fffffffU;
  if (magnitude > 0x7f800000U) {
    return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
  }
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

__device__ __forceinline__ float warp_sum_four(float s0, float s1,
                                               float s2, float s3) {
  const unsigned lane = threadIdx.x % 32;
  const float p0 = __shfl_xor_sync(0xffffffffU, s0, 16);
  const float p1 = __shfl_xor_sync(0xffffffffU, s1, 16);
  const float p2 = __shfl_xor_sync(0xffffffffU, s2, 16);
  const float p3 = __shfl_xor_sync(0xffffffffU, s3, 16);
  const float t0 = __fadd_rn(lane < 16 ? s0 : p2, lane < 16 ? p0 : s2);
  const float t1 = __fadd_rn(lane < 16 ? s1 : p3, lane < 16 ? p1 : s3);
  const float q0 = __shfl_xor_sync(0xffffffffU, t0, 8);
  const float q1 = __shfl_xor_sync(0xffffffffU, t1, 8);
  float sum = __fadd_rn((lane & 8) ? q1 : t0, (lane & 8) ? t1 : q0);
  sum = __fadd_rn(sum, __shfl_down_sync(0xffffffffU, sum, 4, 8));
  sum = __fadd_rn(sum, __shfl_down_sync(0xffffffffU, sum, 2, 8));
  sum = __fadd_rn(sum, __shfl_down_sync(0xffffffffU, sum, 1, 8));
  return sum;
}

// Preserve two original ordered trees in independent sixteen-lane groups.
__device__ __forceinline__ float warp_sum_two(float s0, float s1) {
  const unsigned lane = threadIdx.x & 31;
  const float p0 = __shfl_xor_sync(0xffffffffU, s0, 16);
  const float p1 = __shfl_xor_sync(0xffffffffU, s1, 16);
  float sum = __fadd_rn(lane < 16 ? s0 : p1, lane < 16 ? p0 : s1);
  sum = __fadd_rn(sum, __shfl_down_sync(0xffffffffU, sum, 8, 16));
  sum = __fadd_rn(sum, __shfl_down_sync(0xffffffffU, sum, 4, 16));
  sum = __fadd_rn(sum, __shfl_down_sync(0xffffffffU, sum, 2, 16));
  sum = __fadd_rn(sum, __shfl_down_sync(0xffffffffU, sum, 1, 16));
  return sum;
}

// Keep all six queries resident across independent positions. Each warp
// retains the scalar 256-dimension product/add tree. K is decoded once for
// six heads; no position sum or tensorcore/reassociated dot is introduced.
__device__ __forceinline__ void stage_keys(
    std::uint16_t* destination, const std::uint16_t* keys,
    unsigned int sequence, unsigned int first) {
  if (threadIdx.x < 64) {
  const unsigned int row = threadIdx.x / 32;
  const unsigned int column = (threadIdx.x % 32) * 8;
  const bool valid = first + row < sequence;
  const auto* source = keys + (valid ? (first + row) * 1024 + blockIdx.y * 256 + column : 0);
  const auto dst = static_cast<unsigned int>(__cvta_generic_to_shared(destination + row * 256 + column));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::
               "r"(dst), "l"(source), "r"(valid ? 16 : 0));
  }
  asm volatile("cp.async.commit_group;" ::);
}

template<unsigned M, bool DeviceInputs = false, bool Tree = false>
__global__ void attention_scores_shared_rows_kernel(
    const std::uint16_t* query, const std::uint16_t* keys,
    unsigned int first_sequence, unsigned int rows, float* scores,
    const MtpVerifyDeviceInputs* parameters = nullptr) {
  if constexpr (DeviceInputs) {
    first_sequence = parameters->position + 1;
    if (blockIdx.x * 128 >= first_sequence + M - 1) return;
  }
  const unsigned int warp = threadIdx.x >> 5;
  const unsigned int m = warp % M, position_warp = warp / M;
  const unsigned int tile = blockIdx.x;
  const unsigned int sequence = first_sequence + (Tree && m == M - 1 ? 1 : m);
  query += m * 12288;

  constexpr unsigned int offsets[8] = {0, 128, 64, 192, 32, 160, 96, 224};
  __shared__ float score_tile[M][6][128];
  __shared__ __align__(16) std::uint16_t key_tiles[4][2 * 256];
  const unsigned int lane = threadIdx.x & 31;
  const unsigned int kv = blockIdx.y;
  float queries[6][8];
#pragma unroll
  for (unsigned int h = 0; h < 6; ++h)
#pragma unroll
    for (unsigned int j = 0; j < 8; ++j)
      queries[h][j] = decode_bf16_device(query[(kv * 6 + h) * 256 + lane + offsets[j]]);
#pragma unroll
  for (unsigned int stage = 0; stage < 3; ++stage)
    stage_keys(key_tiles[stage], keys, first_sequence + M - 1, tile * 128 + stage * 2);
  asm volatile("cp.async.wait_group 2;" ::);
  __syncthreads();
  for (unsigned int i = 0; i < 64; ++i) {
    if (i + 3 < 64)
      stage_keys(key_tiles[(i + 3) & 3], keys, first_sequence + M - 1, tile * 128 + (i + 3) * 2);
    const unsigned int column = i * 2 + position_warp;
    const unsigned int position = tile * 128 + column;
    if (position < sequence) {
    float key[8], sums[6];
#pragma unroll
    for (unsigned int j = 0; j < 8; ++j)
      key[j] = decode_bf16_device(Tree && m == M - 1 && position == first_sequence
          ? keys[(first_sequence + M - 2) * 1024 + kv * 256 + lane + offsets[j]]
          : key_tiles[i & 3][position_warp * 256 + lane + offsets[j]]);
#pragma unroll
    for (unsigned int h = 0; h < 6; ++h) {
      float product[8];
#pragma unroll
      for (unsigned int j = 0; j < 8; ++j)
        product[j] = fmaf(queries[h][j], key[j], 0.0f);
      sums[h] = __fadd_rn(
          __fadd_rn(__fadd_rn(product[0], product[1]), __fadd_rn(product[2], product[3])),
          __fadd_rn(__fadd_rn(product[4], product[5]), __fadd_rn(product[6], product[7])));
    }
    const float first = warp_sum_four(sums[0], sums[1], sums[2], sums[3]);
    const float last = warp_sum_two(sums[4], sums[5]);
    if (!(lane & 7)) score_tile[m][lane/8][column] = first * 0.0625f;
    if (!(lane & 15)) score_tile[m][4+lane/16][column] = last * 0.0625f;
    }
    __syncthreads();
    if (i + 3 < 64) asm volatile("cp.async.wait_group 2;" ::);
    else if (i + 2 < 64) asm volatile("cp.async.wait_group 1;" ::);
    else asm volatile("cp.async.wait_group 0;" ::);
    __syncthreads();
  }
  __syncthreads();
  // Publish adjacent positions together instead of one four-byte global
  // store per warp. Invalid tail cells are never read or published.
  for (unsigned int i = threadIdx.x; i < M * 6 * 128; i += 64 * M) {
    const unsigned int row = i / (6 * 128);
    const unsigned int h = (i / 128) % 6;
    const unsigned int column = i % 128;
    const unsigned int position = tile * 128 + column;
    const unsigned int actual = first_sequence + (Tree && row == M - 1 ? 1 : row);
    if (position < actual)
      scores[(row * 24 + kv * 6 + h) *
                 ((first_sequence + M + 2) & ~3U) + position] =
          score_tile[row][h][column];
  }
}

// Same 256-thread max/sum trees and two expf evaluations as scalar softmax.
// Padding owns storage only; actual causal columns alone enter arithmetic.
template<bool DeviceInputs = false, bool Tree = false>
__global__ void aligned_softmax(float* p, unsigned first_sequence,
                                unsigned stride, unsigned rows = 0,
                                const MtpVerifyDeviceInputs* parameters = nullptr) {
  if constexpr (DeviceInputs) {
    first_sequence = parameters->position + 1;
    stride = (first_sequence + rows + 2) & ~3U;
  }
  const unsigned row = blockIdx.x;
  const unsigned columns = first_sequence + (Tree && row / 24 == rows - 1 ? 1 : row / 24);
  p += row * stride;
  __shared__ float partial[256];
  float maximum = -__int_as_float(0x7f800000);
  for (unsigned c = threadIdx.x; c < columns; c += 256)
    maximum = fmaxf(maximum, p[c]);
  partial[threadIdx.x] = maximum;
  __syncthreads();
  for (unsigned d = 128; d; d >>= 1) {
    if (threadIdx.x < d)
      partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + d]);
    __syncthreads();
  }
  maximum = partial[0];
  // Retire every maximum read before any warp publishes its denominator.
  __syncthreads();
  float denominator = 0;
  for (unsigned c = threadIdx.x; c < columns; c += 256)
    denominator += expf(p[c] - maximum);
  partial[threadIdx.x] = denominator;
  __syncthreads();
  for (unsigned d = 128; d; d >>= 1) {
    if (threadIdx.x < d) partial[threadIdx.x] += partial[threadIdx.x + d];
    __syncthreads();
  }
  denominator = partial[0];
  for (unsigned c = threadIdx.x; c < columns; c += 256)
    p[c] = expf(p[c] - maximum) / denominator;
}

// One CTA owns a KV head and 64 output dimensions. Its six consumer warps
// retain two independent outputs per lane; producers load each V tile
// once, while the next tile travels directly from global to shared memory.
// Only operand movement changes. Every accumulator consumes positions in the
// public reference order, including the final partial tile.
__device__ __forceinline__ void ordered_stage(
    std::uint16_t* vs, float* ps, const std::uint16_t* values,
    const float* probabilities, unsigned int sequence, unsigned int first,
    unsigned int dimension, unsigned int probability_stride,
    unsigned int alternate_logical = 0, unsigned int alternate_physical = 0) {
  for (unsigned int i = threadIdx.x; i < 512; i += 192) {
    const unsigned int row = i / 8;
    const unsigned int column = (i % 8) * 8;
    const bool valid = first + row < sequence;
    const unsigned physical = alternate_logical && first + row == alternate_logical
        ? alternate_physical : first + row;
    const auto* src = values + (valid ? physical * 1024 +
        blockIdx.y * 256 + dimension * 64 + column : 0);
    const auto dst = static_cast<unsigned int>(__cvta_generic_to_shared(vs + row * 64 + column));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::
                 "r"(dst), "l"(src), "r"(valid ? 16 : 0));
  }
  // Each head has a 16-byte aligned physical stride. A partial final vector
  // copies only valid probability bytes; no padding value is ever consumed.
  for (unsigned int i = threadIdx.x; i < 96; i += 192) {
    const unsigned int head = i / 16, row = (i % 16) * 4;
    const unsigned int remaining =
        first + row < sequence ? sequence - first - row : 0;
    const unsigned int valid_bytes = (remaining < 4 ? remaining : 4) * 4;
    const auto* src = probabilities + (valid_bytes ?
        (blockIdx.y * 6 + head) * probability_stride + first + row : 0);
    const auto dst = static_cast<unsigned int>(
        __cvta_generic_to_shared(ps + head * 64 + row));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;" ::
                 "r"(dst), "l"(src), "r"(valid_bytes));
  }
  asm volatile("cp.async.commit_group;" ::);
}

template<bool DeviceInputs = false, bool Tree = false>
__global__ void attention_values_ordered_pair_kernel(
    const std::uint16_t* values, const float* probabilities,
    unsigned int first_sequence, unsigned int rows, std::uint16_t* output,
    const MtpVerifyDeviceInputs* parameters = nullptr) {
  if constexpr (DeviceInputs) first_sequence = parameters->position + 1;
  const unsigned int m = blockIdx.x % rows, dimension = blockIdx.x / rows;
  const unsigned int sequence = first_sequence + (Tree && m == rows - 1 ? 1 : m);
  const unsigned alternate_logical = Tree && m == rows - 1 ? first_sequence : 0;
  const unsigned alternate_physical = first_sequence + rows - 2;
  const unsigned int probability_stride = (first_sequence + rows + 2) & ~3U;
  probabilities += m * 24 * probability_stride;
  output += m * 6144;
  __shared__ __align__(16) std::uint16_t vs[4][64 * 64];
  __shared__ __align__(16) float ps[4][6 * 64];
  const unsigned int lane = threadIdx.x % 32;
  const unsigned int head = threadIdx.x / 32;
  float accumulator = 0.0f, second = 0.0f;
  // Three future groups cover operand latency; four physical buffers keep
  // the producer disjoint from the current consumer. S >= 512 guarantees
  // all three initial tiles exist.
#pragma unroll
  for (unsigned int stage = 0; stage < 3; ++stage)
    ordered_stage(vs[stage], ps[stage], values, probabilities, sequence,
                  stage * 64, dimension, probability_stride, alternate_logical, alternate_physical);
  asm volatile("cp.async.wait_group 2;" ::);
  __syncthreads();
  unsigned int buffer = 0;
  for (unsigned int first = 0; first < sequence; first += 64) {
    if (first + 3 * 64 < sequence)
      ordered_stage(vs[(buffer + 3) & 3], ps[(buffer + 3) & 3], values,
                    probabilities, sequence, first + 3 * 64, dimension, probability_stride,
                    alternate_logical, alternate_physical);
    const auto* v = vs[buffer];
    const auto* p = ps[buffer] + head * 64;
    if (first + 64 <= sequence) {
#pragma unroll
      for (unsigned int row = 0; row < 64; ++row) {
        const auto packed = *reinterpret_cast<const unsigned int*>(v + row * 64 + lane * 2);
        const float probability = p[row];
        accumulator = fmaf(probability, __uint_as_float(packed << 16), accumulator);
        second = fmaf(probability, __uint_as_float(packed & 0xffff0000U), second);
      }
    } else {
      for (unsigned int row = 0; row < sequence - first; ++row) {
        const auto packed = *reinterpret_cast<const unsigned int*>(v + row * 64 + lane * 2);
        const float probability = p[row];
        accumulator = fmaf(probability, __uint_as_float(packed << 16), accumulator);
        second = fmaf(probability, __uint_as_float(packed & 0xffff0000U), second);
      }
    }
    __syncthreads();
    // Do not wait for the younger groups while consuming the oldest one.
    // Near the tail, shrink the allowed outstanding count so the next tile
    // is still known complete. Every thread executes the same wait/barrier.
    if (first + 3 * 64 < sequence)
      asm volatile("cp.async.wait_group 2;" ::);
    else if (first + 2 * 64 < sequence)
      asm volatile("cp.async.wait_group 1;" ::);
    else
      asm volatile("cp.async.wait_group 0;" ::);
    __syncthreads();
    buffer = (buffer + 1) & 3;
  }
  const unsigned int destination = (blockIdx.y * 6 + head) * 256 + dimension * 64 + lane * 2;
  output[destination] = encode_bf16_device(accumulator);
  output[destination + 1] = encode_bf16_device(second);
}
template<bool DeviceInputs = false, bool Tree = false>
__global__ void short_attention_rows_kernel(
    const std::uint16_t* query,
    const std::uint16_t* const key_cache,
    const std::uint16_t* const value_cache,
    unsigned first_sequence,
    const float attention_scale,
    const std::uint16_t* gate,
    std::uint16_t* output, const MtpVerifyDeviceInputs* parameters = nullptr) {
  if constexpr (DeviceInputs) first_sequence = parameters->position + 1;
  __shared__ float partial[256];
  __shared__ float probabilities[512];

  const unsigned row = blockIdx.y;
  const unsigned sequence_length = first_sequence + (Tree && row == gridDim.y - 1 ? 1 : row);
  query += row * 12288;
  gate += row * 12288;
  output += row * 6144;
  const std::size_t query_head = static_cast<std::size_t>(blockIdx.x);
  const std::size_t queries_per_kv =
      24 / 4;
  const std::size_t kv_head = query_head / queries_per_kv;
  const std::size_t query_offset =
      query_head * 256;

  const unsigned int warp = threadIdx.x >> 5U;
  const unsigned int lane = threadIdx.x & 31U;
  for (std::size_t position = warp; position < sequence_length;
       position += 8) {
    const std::size_t key_offset =
        ((Tree && row == gridDim.y - 1 && position == first_sequence
            ? first_sequence + gridDim.y - 2 : position) * 4 + kv_head) * 256;

    float sum = fmaf(decode_bf16_device(query[query_offset + lane]),
                     decode_bf16_device(key_cache[key_offset + lane]), 0.0F);
    float rhs =
        fmaf(decode_bf16_device(query[query_offset + lane + 128U]),
             decode_bf16_device(key_cache[key_offset + lane + 128U]), 0.0F);
    sum += rhs;

    float sibling =
        fmaf(decode_bf16_device(query[query_offset + lane + 64U]),
             decode_bf16_device(key_cache[key_offset + lane + 64U]), 0.0F);
    rhs = fmaf(decode_bf16_device(query[query_offset + lane + 192U]),
               decode_bf16_device(key_cache[key_offset + lane + 192U]),
               0.0F);
    sibling += rhs;
    sum += sibling;

    sibling =
        fmaf(decode_bf16_device(query[query_offset + lane + 32U]),
             decode_bf16_device(key_cache[key_offset + lane + 32U]), 0.0F);
    rhs = fmaf(decode_bf16_device(query[query_offset + lane + 160U]),
               decode_bf16_device(key_cache[key_offset + lane + 160U]),
               0.0F);
    sibling += rhs;
    float upper =
        fmaf(decode_bf16_device(query[query_offset + lane + 96U]),
             decode_bf16_device(key_cache[key_offset + lane + 96U]), 0.0F);
    rhs = fmaf(decode_bf16_device(query[query_offset + lane + 224U]),
               decode_bf16_device(key_cache[key_offset + lane + 224U]),
               0.0F);
    upper += rhs;
    sibling += upper;
    sum += sibling;

#pragma unroll
    for (unsigned int stride = 16U; stride != 0U; stride >>= 1U) {
      rhs = __shfl_down_sync(0xffffffffU, sum, stride);
      if (lane < stride) {
        sum += rhs;
      }
    }
    if (lane == 0U) {
      probabilities[position] = sum * attention_scale;
    }
  }
  __syncthreads();

  float maximum = -__int_as_float(0x7f800000);
  for (std::size_t position = threadIdx.x; position < sequence_length;
       position += blockDim.x) {
    maximum = fmaxf(maximum, probabilities[position]);
  }
  partial[threadIdx.x] = maximum;
  __syncthreads();
  for (unsigned int stride = 256 / 2U; stride != 0U; stride >>= 1U) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] =
          fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  maximum = partial[0];
  // All readers must retire before partial is reused for the sum.
  __syncthreads();

  float denominator = 0.0F;
  for (std::size_t position = threadIdx.x; position < sequence_length;
       position += blockDim.x) {
    denominator += expf(probabilities[position] - maximum);
  }
  partial[threadIdx.x] = denominator;
  __syncthreads();
  for (unsigned int stride = 256 / 2U; stride != 0U; stride >>= 1U) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  denominator = partial[0];

  for (std::size_t position = threadIdx.x; position < sequence_length;
       position += blockDim.x) {
    const float probability =
        expf(probabilities[position] - maximum) / denominator;
    probabilities[position] = probability;
  }
  __syncthreads();

  for (std::size_t dimension = threadIdx.x;
       dimension < 256;
       dimension += blockDim.x) {
    float value = 0.0F;
    for (std::size_t position = 0U; position < sequence_length; ++position) {
      const std::size_t value_offset =
          ((Tree && row == gridDim.y - 1 && position == first_sequence
            ? first_sequence + gridDim.y - 2 : position) * 4 + kv_head) * 256;
      value = fmaf(probabilities[position],
                   decode_bf16_device(
                       value_cache[value_offset + dimension]),
                   value);
    }
    const std::uint16_t encoded_value = encode_bf16_device(value);
    const float gate_value =
        decode_bf16_device(gate[query_offset + dimension]);
    const float sigmoid =
        gate_value >= 0.0F
            ? 1.0F / (1.0F + expf(-gate_value))
            : expf(gate_value) / (1.0F + expf(gate_value));
    output[query_offset + dimension] = encode_bf16_device(
        decode_bf16_device(encoded_value) * sigmoid);
  }
}

struct Span { std::uintptr_t begin; std::size_t bytes; };
bool valid_spans(const std::array<Span, 5>& spans) noexcept {
  for (std::size_t i = 0; i < spans.size(); ++i) {
    const auto a = spans[i];
    if (!a.begin || a.begin > std::numeric_limits<std::uintptr_t>::max() - a.bytes)
      return false;
    for (std::size_t j = 0; j < i; ++j) {
      const auto b = spans[j];
      if (a.begin < b.begin + b.bytes && b.begin < a.begin + a.bytes) return false;
    }
  }
  return true;
}
}  // namespace


// Private batch interface: query rows have [Q6144, gate6144] stride; each
// probability row uses its actual causal head stride inside a fixed slot.
int launch_mtp_verify_attention(const std::uint16_t* query,
    const std::uint16_t* key, const std::uint16_t* value,
    unsigned first_sequence, unsigned rows, void* workspace,
    std::size_t workspace_bytes, std::uint16_t* output, void* opaque) noexcept {
  if (rows < 2 || rows > 4 || first_sequence < fused_decode::kMinimumSequence ||
      first_sequence > fused_decode::kMaximumSequence - (rows-1))
    return cudaErrorInvalidValue;
  const std::size_t last = first_sequence + rows - 1;
  const unsigned int probability_stride = (last + 3) & ~std::size_t(3);
  const std::size_t bytes = rows * 24 * probability_stride * sizeof(float);
  if (workspace_bytes < bytes ||
      reinterpret_cast<std::uintptr_t>(workspace) % 16 ||
      reinterpret_cast<std::uintptr_t>(query) % 16 ||
      reinterpret_cast<std::uintptr_t>(key) % 16 ||
      reinterpret_cast<std::uintptr_t>(value) % 16 ||
      reinterpret_cast<std::uintptr_t>(output) % 16 ||
      !valid_spans({{{reinterpret_cast<std::uintptr_t>(query), rows * 12288 * 2},
                    {reinterpret_cast<std::uintptr_t>(key), last * 1024 * 2},
                    {reinterpret_cast<std::uintptr_t>(value), last * 1024 * 2},
                    {reinterpret_cast<std::uintptr_t>(workspace), bytes},
                    {reinterpret_cast<std::uintptr_t>(output), rows * 6144 * 2}}}))
    return cudaErrorInvalidValue;
  (void)cudaGetLastError();
  const auto stream = static_cast<cudaStream_t>(opaque);
  auto* probabilities = static_cast<float*>(workspace);
  const dim3 score_grid((last + 127)/128,4);
  if (rows == 2)
    attention_scores_shared_rows_kernel<2><<<score_grid,128,0,stream>>>(
        query,key,first_sequence,rows,probabilities);
  else if (rows == 3)
    attention_scores_shared_rows_kernel<3><<<score_grid,192,0,stream>>>(
        query,key,first_sequence,rows,probabilities);
  else
    attention_scores_shared_rows_kernel<4><<<score_grid,256,0,stream>>>(
        query,key,first_sequence,rows,probabilities);
  auto status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  aligned_softmax<<<24 * rows, 256, 0, stream>>>(
      probabilities, first_sequence, probability_stride);
  status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  attention_values_ordered_pair_kernel<<<dim3(4*rows,4),192,0,stream>>>(
      value,probabilities,first_sequence,rows,output);
  return cudaGetLastError();
}
} // namespace q3x::runtime::mtp_detail

namespace q3x::runtime::mtp_detail {
int launch_mtp_short_attention(const std::uint16_t* query,
    const std::uint16_t* keys, const std::uint16_t* values,
    unsigned first_sequence, unsigned rows, std::uint16_t* output,
    void* opaque) noexcept {
  if (rows < 2 || rows > 4 || first_sequence < 65 ||
      first_sequence > 512 - rows) return cudaErrorInvalidValue;
  const std::size_t last = first_sequence + rows - 1;
  const Span spans[] = {{reinterpret_cast<std::uintptr_t>(query), rows * 12288ULL * 2},
    {reinterpret_cast<std::uintptr_t>(keys), last * 1024 * 2},
    {reinterpret_cast<std::uintptr_t>(values), last * 1024 * 2},
    {reinterpret_cast<std::uintptr_t>(output), rows * 6144ULL * 2}};
  for (unsigned i = 0; i < 4; ++i) {
    const auto a = spans[i];
    if (!a.begin || a.begin % 16 || a.begin > std::numeric_limits<std::uintptr_t>::max() - a.bytes)
      return cudaErrorInvalidValue;
    for (unsigned j = 0; j < i; ++j)
      if (a.begin < spans[j].begin + spans[j].bytes && spans[j].begin < a.begin + a.bytes)
        return cudaErrorInvalidValue;
  }
  (void)cudaGetLastError();
  short_attention_rows_kernel<<<dim3(24,rows),256,0,static_cast<cudaStream_t>(opaque)>>>(
      query, keys, values, first_sequence, .0625F, query + 6144, output);
  return cudaGetLastError();
}
} // namespace q3x::runtime::mtp_detail

namespace q3x::runtime::mtp_detail {
template<bool Tree>
int launch_graph_attention_impl(const std::uint16_t* query,
    const std::uint16_t* key, const std::uint16_t* value,
    const MtpVerifyDeviceInputs* parameters, unsigned maximum_first_sequence,
    unsigned rows, void* workspace, std::size_t workspace_bytes,
    std::uint16_t* output, void* opaque) noexcept {
  if (rows < (Tree ? 3U : 2U) || rows > 4 || maximum_first_sequence < 65 ||
      maximum_first_sequence > fused_decode::kMaximumSequence - (rows - 1) ||
      !parameters || reinterpret_cast<std::uintptr_t>(parameters) % 16)
    return cudaErrorInvalidValue;
  const bool short_path = maximum_first_sequence < 512;
  if (short_path && maximum_first_sequence > 512 - rows)
    return cudaErrorInvalidValue;
  const std::size_t last = maximum_first_sequence + rows - 1;
  const unsigned stride = (last + 3) & ~std::size_t(3);
  const std::size_t bytes = rows * 24ULL * stride * sizeof(float);
  const Span spans[] = {
    {reinterpret_cast<std::uintptr_t>(query), rows * 12288ULL * 2},
    {reinterpret_cast<std::uintptr_t>(key), last * 1024 * 2},
    {reinterpret_cast<std::uintptr_t>(value), last * 1024 * 2},
    {reinterpret_cast<std::uintptr_t>(output), rows * 6144ULL * 2},
    {reinterpret_cast<std::uintptr_t>(parameters), sizeof(*parameters)},
    {reinterpret_cast<std::uintptr_t>(workspace), bytes}};
  if (!short_path && workspace_bytes < bytes) return cudaErrorInvalidValue;
  for (unsigned i = 0; i < (short_path ? 5U : 6U); ++i) {
    const auto a = spans[i];
    if (!a.begin || a.begin % 16 ||
        a.begin > std::numeric_limits<std::uintptr_t>::max() - a.bytes)
      return cudaErrorInvalidValue;
    for (unsigned j = 0; j < i; ++j)
      if (a.begin < spans[j].begin + spans[j].bytes &&
          spans[j].begin < a.begin + a.bytes) return cudaErrorInvalidValue;
  }
  (void)cudaGetLastError();
  const auto stream = static_cast<cudaStream_t>(opaque);
  if (short_path) {
    short_attention_rows_kernel<true, Tree><<<dim3(24, rows), 256, 0, stream>>>(
        query, key, value, 0, .0625F, query + 6144, output, parameters);
    return cudaGetLastError();
  }
  auto* probabilities = static_cast<float*>(workspace);
  const dim3 grid((last + 127) / 128, 4);
  if (rows == 2)
    attention_scores_shared_rows_kernel<2, true, Tree><<<grid, 128, 0, stream>>>(
        query, key, 0, rows, probabilities, parameters);
  else if (rows == 3)
    attention_scores_shared_rows_kernel<3, true, Tree><<<grid, 192, 0, stream>>>(
        query, key, 0, rows, probabilities, parameters);
  else
    attention_scores_shared_rows_kernel<4, true, Tree><<<grid, 256, 0, stream>>>(
        query, key, 0, rows, probabilities, parameters);
  auto status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  aligned_softmax<true, Tree><<<24 * rows, 256, 0, stream>>>(
      probabilities, 0, 0, rows, parameters);
  status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  attention_values_ordered_pair_kernel<true, Tree><<<dim3(4 * rows, 4), 192, 0, stream>>>(
      value, probabilities, 0, rows, output, parameters);
  return cudaGetLastError();
}

int launch_mtp_graph_attention(const std::uint16_t* query,
    const std::uint16_t* key, const std::uint16_t* value,
    const MtpVerifyDeviceInputs* parameters, unsigned maximum_first_sequence,
    unsigned rows, void* workspace, std::size_t workspace_bytes,
    std::uint16_t* output, void* opaque) noexcept {
  return launch_graph_attention_impl<false>(query, key, value, parameters,
      maximum_first_sequence, rows, workspace, workspace_bytes, output, opaque);
}

int launch_mtp_tree_graph_attention(const std::uint16_t* query,
    const std::uint16_t* key, const std::uint16_t* value,
    const MtpVerifyDeviceInputs* parameters, unsigned maximum_first_sequence,
    unsigned rows, void* workspace, std::size_t workspace_bytes,
    std::uint16_t* output, void* opaque) noexcept {
  return launch_graph_attention_impl<true>(query, key, value, parameters,
      maximum_first_sequence, rows, workspace, workspace_bytes, output, opaque);
}

bool is_mtp_graph_qk_kernel(const void* function) noexcept {
  return function == reinterpret_cast<const void*>(attention_scores_shared_rows_kernel<2, true>) ||
         function == reinterpret_cast<const void*>(attention_scores_shared_rows_kernel<3, true>) ||
         function == reinterpret_cast<const void*>(attention_scores_shared_rows_kernel<4, true>);
}
}  // namespace q3x::runtime::mtp_detail

namespace q3x::runtime::mtp_detail {
namespace {
__global__ void mtp_graph_embedding_kernel(const std::uint16_t* embedding,
    const MtpVerifyDeviceInputs* parameters, std::uint16_t* output) {
  const unsigned column = blockIdx.x * 256 + threadIdx.x;
  const unsigned row = blockIdx.y;
  if (column < 5120)
    output[row * 5120 + column] =
        embedding[std::size_t(parameters->token_ids[row]) * 5120 + column];
}
__global__ void mtp_graph_preprocess_kernel(
    const std::uint16_t* const interleaved_q_gate,
    std::uint16_t* const key,
    const std::uint16_t* const q_weight,
    const std::uint16_t* const k_weight,
    const float epsilon,
    std::uint16_t* const query_output,
    std::uint16_t* const gate_output,
    const float* const cosines,
    const float* const sines,
    const MtpVerifyDeviceInputs* parameters,
    const std::uint16_t* value_stage,
    std::uint16_t* key_cache, std::uint16_t* value_cache) {
  constexpr unsigned kFullPreprocessQueryHeads = 24, kFullPreprocessKvHeads = 4;
  constexpr unsigned kFullPreprocessHeadDimension = 256, kThreads = 256;
  constexpr unsigned kQwenRotaryDimension = 64;
  const std::size_t first_position = parameters->position;
  constexpr std::size_t kCombinedHeads =
      kFullPreprocessQueryHeads + kFullPreprocessKvHeads;
  __shared__ float partial[kThreads];

  const std::size_t combined_head = static_cast<std::size_t>(blockIdx.x);
  const std::size_t token = combined_head / kCombinedHeads;
  const std::size_t token_head = combined_head - token * kCombinedHeads;
  const bool is_query = token_head < kFullPreprocessQueryHeads;
  const std::size_t head =
      is_query ? token_head : token_head - kFullPreprocessQueryHeads;
  const std::size_t dimension = threadIdx.x;

  std::size_t packed_offset = 0U;
  const std::uint16_t* weight = nullptr;
  float value = 0.0F;
  if (is_query) {
    packed_offset =
        token * 12288 + head * kFullPreprocessHeadDimension + dimension;
    const std::size_t interleaved_offset =
        (token * kFullPreprocessQueryHeads + head) *
            (2U * kFullPreprocessHeadDimension) +
        dimension;
    value = decode_bf16_device(interleaved_q_gate[interleaved_offset]);
    gate_output[packed_offset] =
        interleaved_q_gate[interleaved_offset +
                           kFullPreprocessHeadDimension];
    weight = q_weight;
  } else {
    packed_offset =
        (token * kFullPreprocessKvHeads + head) *
            kFullPreprocessHeadDimension +
        dimension;
    value = decode_bf16_device(key[packed_offset]);
    weight = k_weight;
  }

  float sum = 0.0F;
  sum = fmaf(value, value, sum);
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (unsigned int stride = kThreads / 2U; stride != 0U; stride >>= 1U) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inverse_rms =
      rsqrtf(partial[0] /
                 static_cast<float>(kFullPreprocessHeadDimension) +
             epsilon);
  const float gamma = decode_bf16_device(weight[dimension]) + 1.0F;
  const std::uint16_t normalized =
      encode_bf16_device(value * inverse_rms * gamma);
  std::uint16_t* const normalized_output = is_query ? query_output : key;
  normalized_output[packed_offset] = normalized;

  constexpr std::size_t kHalfRotary = kQwenRotaryDimension / 2U;
  __syncthreads();
  if (dimension < kHalfRotary) {
    std::uint16_t* const head_output =
        normalized_output + packed_offset - dimension;
    const std::size_t table_offset =
        (first_position + token) * kHalfRotary + dimension;
    const float cosine = cosines[table_offset];
    const float sine = sines[table_offset];
    const float first = decode_bf16_device(head_output[dimension]);
    const float second =
        decode_bf16_device(head_output[dimension + kHalfRotary]);
    // Spell out which product is rounded before the FFMA. This is the exact
    // instruction order emitted for qk_partial_neox_rope_tile above.
    const float rotated_first = fmaf(first, cosine, -(second * sine));
    const float rotated_second = fmaf(second, cosine, first * sine);
    head_output[dimension] = encode_bf16_device(rotated_first);
    head_output[dimension + kHalfRotary] =
        encode_bf16_device(rotated_second);
  }
  // Each K head owns its complete cache destination. Retire all RoPE writes
  // before copying that head; no cross-CTA dependency or rounding is added.
  __syncthreads();
  if (!is_query) {
    const std::size_t destination = (first_position + token) * 1024 +
        head * kFullPreprocessHeadDimension + dimension;
    key_cache[destination] = key[packed_offset];
    value_cache[destination] = value_stage[packed_offset];
  }
}


}  // namespace
int launch_mtp_graph_embedding(const std::uint16_t* embedding,
    const MtpVerifyDeviceInputs* parameters, unsigned rows,
    std::uint16_t* output, void* opaque) noexcept {
  if (!embedding || !parameters || !output || rows < 2 || rows > 4)
    return cudaErrorInvalidValue;
  mtp_graph_embedding_kernel<<<dim3(20, rows), 256, 0,
      static_cast<cudaStream_t>(opaque)>>>(embedding, parameters, output);
  return cudaGetLastError();
}
int launch_mtp_graph_preprocess(const std::uint16_t* q_gate,
    std::uint16_t* key_stage, const std::uint16_t* value_stage,
    const std::uint16_t* q_norm, const std::uint16_t* k_norm,
    const float* cosines, const float* sines,
    const MtpVerifyDeviceInputs* parameters, unsigned rows,
    std::uint16_t* query_gate, std::uint16_t* key_cache,
    std::uint16_t* value_cache, void* opaque) noexcept {
  if (!q_gate || !key_stage || !value_stage || !q_norm || !k_norm ||
      !cosines || !sines || !parameters || !query_gate || !key_cache ||
      !value_cache || rows < 2 || rows > 4) return cudaErrorInvalidValue;
  mtp_graph_preprocess_kernel<<<28 * rows, 256, 0,
      static_cast<cudaStream_t>(opaque)>>>(q_gate, key_stage, q_norm, k_norm,
      1.e-6F, query_gate, query_gate + 6144, cosines, sines, parameters,
      value_stage, key_cache, value_cache);
  return cudaGetLastError();
}
}  // namespace q3x::runtime::mtp_detail
