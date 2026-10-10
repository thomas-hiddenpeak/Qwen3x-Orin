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
  const unsigned int row = threadIdx.x / 32;
  const unsigned int column = (threadIdx.x % 32) * 8;
  const bool valid = first + row < sequence;
  const auto* source = keys + (valid ? (first + row) * 1024 + blockIdx.y * 256 + column : 0);
  const auto dst = static_cast<unsigned int>(__cvta_generic_to_shared(destination + row * 256 + column));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::
               "r"(dst), "l"(source), "r"(valid ? 16 : 0));
  asm volatile("cp.async.commit_group;" ::);
}

__global__ void attention_scores_grouped_async_kernel(
    const std::uint16_t* query, const std::uint16_t* keys,
    unsigned int first_sequence, unsigned int rows, float* scores) {
  const unsigned int m = blockIdx.x % rows, tile = blockIdx.x / rows;
  const unsigned int sequence = first_sequence + m;
  query += m * 12288;
  scores += m * 24 * (first_sequence + rows - 1);
  constexpr unsigned int offsets[8] = {0, 128, 64, 192, 32, 160, 96, 224};
  __shared__ float score_tile[6][128];
  __shared__ __align__(16) std::uint16_t key_tiles[4][8 * 256];
  const unsigned int lane = threadIdx.x & 31;
  const unsigned int warp = threadIdx.x >> 5;
  const unsigned int kv = blockIdx.y;
  float queries[6][8];
#pragma unroll
  for (unsigned int h = 0; h < 6; ++h)
#pragma unroll
    for (unsigned int j = 0; j < 8; ++j)
      queries[h][j] = decode_bf16_device(query[(kv * 6 + h) * 256 + lane + offsets[j]]);
#pragma unroll
  for (unsigned int stage = 0; stage < 3; ++stage)
    stage_keys(key_tiles[stage], keys, sequence, tile * 128 + stage * 8);
  asm volatile("cp.async.wait_group 2;" ::);
  __syncthreads();
  for (unsigned int i = 0; i < 16; ++i) {
    if (i + 3 < 16)
      stage_keys(key_tiles[(i + 3) & 3], keys, sequence, tile * 128 + (i + 3) * 8);
    const unsigned int column = i * 8 + warp;
    const unsigned int position = tile * 128 + column;
    if (position < sequence) {
    float key[8], sums[6];
#pragma unroll
    for (unsigned int j = 0; j < 8; ++j)
      key[j] = decode_bf16_device(key_tiles[i & 3][warp * 256 + lane + offsets[j]]);
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
    if (!(lane & 7)) score_tile[lane/8][column] = first * 0.0625f;
    if (!(lane & 15)) score_tile[4+lane/16][column] = last * 0.0625f;
    }
    __syncthreads();
    if (i + 3 < 16) asm volatile("cp.async.wait_group 2;" ::);
    else if (i + 2 < 16) asm volatile("cp.async.wait_group 1;" ::);
    else asm volatile("cp.async.wait_group 0;" ::);
    __syncthreads();
  }
  __syncthreads();
  // Publish adjacent positions together instead of one four-byte global
  // store per warp. Invalid tail cells are never read or published.
  for (unsigned int i = threadIdx.x; i < 6 * 128; i += 256) {
    const unsigned int h = i / 128;
    const unsigned int column = i % 128;
    const unsigned int position = tile * 128 + column;
    if (position < sequence)
      scores[(kv * 6 + h) * sequence + position] = score_tile[h][column];
  }
}

// One CTA owns a KV head and 64 output dimensions. Its six consumer warps
// retain two independent outputs per lane; producers load each V tile
// once, while the next tile travels directly from global to shared memory.
// Only operand movement changes. Every accumulator consumes positions in the
// public reference order, including the final partial tile.
__device__ __forceinline__ void ordered_stage(
    std::uint16_t* vs, float* ps, const std::uint16_t* values,
    const float* probabilities, unsigned int sequence, unsigned int first, unsigned int dimension) {
  for (unsigned int i = threadIdx.x; i < 512; i += 192) {
    const unsigned int row = i / 8;
    const unsigned int column = (i % 8) * 8;
    const bool valid = first + row < sequence;
    const auto* src = values + (valid ? (first + row) * 1024 +
        blockIdx.y * 256 + dimension * 64 + column : 0);
    const auto dst = static_cast<unsigned int>(__cvta_generic_to_shared(vs + row * 64 + column));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::
                 "r"(dst), "l"(src), "r"(valid ? 16 : 0));
  }
  for (unsigned int i = threadIdx.x; i < 384; i += 192) {
    const unsigned int head = i / 64;
    const unsigned int row = i % 64;
    const bool valid = first + row < sequence;
    const auto* src = probabilities + (valid ?
        (blockIdx.y * 6 + head) * sequence + first + row : 0);
    const auto dst = static_cast<unsigned int>(__cvta_generic_to_shared(ps + i));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::
                 "r"(dst), "l"(src), "r"(valid ? 4 : 0));
  }
  asm volatile("cp.async.commit_group;" ::);
}

__global__ void attention_values_ordered_pair_kernel(
    const std::uint16_t* values, const float* probabilities,
    unsigned int first_sequence, unsigned int rows, std::uint16_t* output) {
  const unsigned int m = blockIdx.x % rows, dimension = blockIdx.x / rows;
  const unsigned int sequence = first_sequence + m;
  probabilities += m * 24 * (first_sequence + rows - 1);
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
    ordered_stage(vs[stage], ps[stage], values, probabilities, sequence, stage * 64, dimension);
  asm volatile("cp.async.wait_group 2;" ::);
  __syncthreads();
  unsigned int buffer = 0;
  for (unsigned int first = 0; first < sequence; first += 64) {
    if (first + 3 * 64 < sequence)
      ordered_stage(vs[(buffer + 3) & 3], ps[(buffer + 3) & 3], values,
                    probabilities, sequence, first + 3 * 64, dimension);
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
  const std::size_t bytes = rows * 24 * last * sizeof(float);
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
  attention_scores_grouped_async_kernel
      <<<dim3(((last + 127)/128)*rows,4),256,0,stream>>>(
          query,key,first_sequence,rows,probabilities);
  auto status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  for (unsigned m=0;m<rows;++m) {
    auto* p = probabilities + m * 24 * last;
    const int code = launch_softmax_reference_cuda(p,24,first_sequence+m,p,opaque);
    if (code) return code;
  }
  attention_values_ordered_pair_kernel<<<dim3(4*rows,4),192,0,stream>>>(
      value,probabilities,first_sequence,rows,output);
  return cudaGetLastError();
}
} // namespace q3x::runtime::mtp_detail
