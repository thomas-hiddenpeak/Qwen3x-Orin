/*
 * Copyright 2026 Qwen3x-Orin contributors.
 * Portions of the grouped-query Prefill dataflow and fragment-to-output
 * mapping are adapted from FlashInfer, Copyright 2023-2026 FlashInfer
 * community and Copyright 2025-2026 NVIDIA.
 *
 * Licensed under the Apache License, Version 2.0.  This file is a
 * fixed-model, SM87-specific modification and is not part of FlashInfer.
 */
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace q3x::runtime {
namespace {

constexpr unsigned int kBulkGqaQueryHeads = 24U;
constexpr unsigned int kBulkGqaKvHeads = 4U;
constexpr unsigned int kBulkGqaQueriesPerKv =
    kBulkGqaQueryHeads / kBulkGqaKvHeads;
constexpr unsigned int kBulkGqaHeadDimension = 256U;
constexpr unsigned int kBulkGqaTensorCoreQueryTile = 16U;
constexpr unsigned int kBulkGqaTensorCoreKvTile = 16U;
constexpr unsigned int kBulkGqaRegisterWarpCount = 4U;
constexpr unsigned int kBulkGqaRegisterThreads =
    kBulkGqaRegisterWarpCount * 32U;
constexpr float kBulkGqaAttentionScale = 1.0F / 16.0F;
constexpr unsigned int kBulkGqaC512TokenCount = 512U;
constexpr unsigned int kBulkGqaPackedQueryCount =
    kBulkGqaC512TokenCount * kBulkGqaQueriesPerKv;
constexpr unsigned int kBulkGqaGroupQueryTile = 64U;
constexpr unsigned int kBulkGqaGroupKvTile = 32U;
constexpr unsigned int kBulkGqaGroupGridX =
    kBulkGqaPackedQueryCount / kBulkGqaGroupQueryTile;

static_assert(kBulkGqaQueriesPerKv == 6U);
static_assert(kBulkGqaRegisterThreads == 128U);
static_assert(kBulkGqaPackedQueryCount == 3'072U);
static_assert(kBulkGqaGroupGridX == 48U);

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

// SM80/SM87 FlashAttention-style register pipeline for the one production
// shape P0/C512/H24/KV4/D256. Four warps cooperatively own one Q16 tile and
// one query head. Each warp walks an interleaved K/V16 subsequence, retaining
// its complete FP32 online-softmax (m, d, O) state in registers. K/V are
// transferred with the same K-current/V-current/K-next/V-next cp.async
// schedule used by FlashInfer's Ampere Prefill mainloop. The four independent
// warp states are merged once after the mainloop; unlike the predecessor,
// there is no per-K/V-tile 16x256 FP32 shared-memory read/scale/write cycle.
struct BulkGqaRegisterLoadStorage {
  std::uint16_t
      query[kBulkGqaTensorCoreQueryTile * kBulkGqaHeadDimension];
  std::uint16_t
      key[kBulkGqaRegisterWarpCount * kBulkGqaTensorCoreKvTile *
          kBulkGqaHeadDimension];
  std::uint16_t
      value[kBulkGqaRegisterWarpCount * kBulkGqaTensorCoreKvTile *
            kBulkGqaHeadDimension];
};

struct BulkGqaRegisterMergeStorage {
  float partial_output
      [kBulkGqaRegisterWarpCount * kBulkGqaTensorCoreQueryTile *
       kBulkGqaHeadDimension];
  float maxima[kBulkGqaRegisterWarpCount *
               kBulkGqaTensorCoreQueryTile];
  float denominators[kBulkGqaRegisterWarpCount *
                     kBulkGqaTensorCoreQueryTile];
};

union BulkGqaRegisterSharedStorage {
  BulkGqaRegisterLoadStorage load;
  BulkGqaRegisterMergeStorage merge;
};

static_assert(sizeof(BulkGqaRegisterLoadStorage) == 72U * 1024U);
static_assert(sizeof(BulkGqaRegisterMergeStorage) <
              sizeof(BulkGqaRegisterLoadStorage));

__device__ __forceinline__ void bulk_gqa_cp_async_cg_16(
    void* const shared_destination, const void* const global_source) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const unsigned int shared_address =
      static_cast<unsigned int>(__cvta_generic_to_shared(shared_destination));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :
               : "r"(shared_address), "l"(global_source)
               : "memory");
#else
  *reinterpret_cast<uint4*>(shared_destination) =
      *reinterpret_cast<const uint4*>(global_source);
#endif
}

__device__ __forceinline__ void bulk_gqa_cp_async_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;" ::: "memory");
#endif
}

__device__ __forceinline__ void bulk_gqa_cp_async_wait_group_1() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group 1;" ::: "memory");
#endif
}

__device__ __forceinline__ void bulk_gqa_cp_async_wait_group_0() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group 0;" ::: "memory");
#endif
}

__device__ __forceinline__ float bulk_gqa_fast_exp(const float value) {
#if defined(__CUDA_ARCH__)
  float result = 0.0F;
  asm("ex2.approx.f32 %0, %1;" : "=f"(result)
      : "f"(value * 1.4426950408889634F));
  return result;
#else
  return expf(value);
#endif
}

__device__ __forceinline__ std::uint16_t bulk_gqa_apply_sigmoid_gate(
    const float attention, const std::uint16_t gate_bits) {
  const std::uint16_t rounded_attention = encode_bf16_device(attention);
  const float gate_value = decode_bf16_device(gate_bits);
  const float gate_exp = bulk_gqa_fast_exp(-fabsf(gate_value));
  const float sigmoid = gate_value >= 0.0F
                            ? 1.0F / (1.0F + gate_exp)
                            : gate_exp / (1.0F + gate_exp);
  return encode_bf16_device(
      decode_bf16_device(rounded_attention) * sigmoid);
}

__device__ __forceinline__ void bulk_gqa_stage_register_kv_tile(
    std::uint16_t* const shared,
    const std::uint16_t* const global,
    const unsigned int kv_head,
    const unsigned int tile_start,
    const bool valid_tile,
    const unsigned int warp,
    const unsigned int lane) {
  constexpr unsigned int kVectorsPerHead =
      kBulkGqaHeadDimension * sizeof(std::uint16_t) / sizeof(uint4);
  constexpr unsigned int kVectorsPerTile =
      kBulkGqaTensorCoreKvTile * kVectorsPerHead;
  static_assert(kVectorsPerHead == 32U);
  static_assert(kVectorsPerTile == 512U);
  if (valid_tile) {
#pragma unroll
    for (unsigned int vector = lane; vector < kVectorsPerTile;
         vector += 32U) {
      const unsigned int local_position = vector / kVectorsPerHead;
      const unsigned int vector_in_head =
          vector - local_position * kVectorsPerHead;
      const std::size_t global_vector =
          (static_cast<std::size_t>(tile_start + local_position) *
               kBulkGqaKvHeads +
           kv_head) *
              kVectorsPerHead +
          vector_in_head;
      const unsigned int shared_vector =
          warp * kVectorsPerTile + vector;
      bulk_gqa_cp_async_cg_16(
          shared + shared_vector * (sizeof(uint4) / sizeof(std::uint16_t)),
          reinterpret_cast<const uint4*>(global) + global_vector);
    }
  }
  bulk_gqa_cp_async_commit();
}

__global__ __launch_bounds__(kBulkGqaRegisterThreads, 1)
void bulk_causal_gqa_sigmoid_gate_24_4_256_c512_register_pipeline_kernel(
    const std::uint16_t* const query,
    const std::uint16_t* const key_cache,
    const std::uint16_t* const value_cache,
    const std::uint16_t* const gate,
    std::uint16_t* const output) {
  // CUDA guarantees a suitably aligned base for dynamic shared memory.  Avoid
  // forcing a module-wide static-shared alignment that perturbs unrelated
  // kernel resource contracts in this translation unit.
  extern __shared__ unsigned char dynamic_shared[];
  auto& storage =
      *reinterpret_cast<BulkGqaRegisterSharedStorage*>(dynamic_shared);

  constexpr unsigned int kQueryVectors =
      kBulkGqaTensorCoreQueryTile * kBulkGqaHeadDimension *
      sizeof(std::uint16_t) / sizeof(uint4);
  constexpr unsigned int kOutputFragments = kBulkGqaHeadDimension / 16U;
  constexpr unsigned int kMatrixElements =
      kBulkGqaTensorCoreQueryTile * kBulkGqaHeadDimension;
  static_assert(kQueryVectors == 512U);
  static_assert(kOutputFragments == 16U);

  const unsigned int thread = threadIdx.x;
  const unsigned int warp = thread >> 5U;
  const unsigned int lane = thread & 31U;
  const unsigned int query_head = blockIdx.y;
  const unsigned int kv_head = query_head / kBulkGqaQueriesPerKv;
  const unsigned int first_query_token =
      blockIdx.x * kBulkGqaTensorCoreQueryTile;
  const unsigned int causal_kv_length =
      first_query_token + kBulkGqaTensorCoreQueryTile;
  const unsigned int iteration_count =
      (causal_kv_length +
       kBulkGqaRegisterWarpCount * kBulkGqaTensorCoreKvTile - 1U) /
      (kBulkGqaRegisterWarpCount * kBulkGqaTensorCoreKvTile);

  auto* const shared_query_words =
      reinterpret_cast<std::uint32_t*>(storage.load.query);
  for (unsigned int vector = thread; vector < kQueryVectors;
       vector += kBulkGqaRegisterThreads) {
    const unsigned int local_query = vector / 32U;
    const unsigned int vector_in_head = vector - local_query * 32U;
    const std::size_t global_vector =
        (static_cast<std::size_t>(first_query_token + local_query) *
             kBulkGqaQueryHeads +
         query_head) *
            32U +
        vector_in_head;
    const uint4 packed_query =
        reinterpret_cast<const uint4*>(query)[global_vector];
    const unsigned int first_word = vector * 4U;
    shared_query_words[first_word] = packed_query.x;
    shared_query_words[first_word + 1U] = packed_query.y;
    shared_query_words[first_word + 2U] = packed_query.z;
    shared_query_words[first_word + 3U] = packed_query.w;
  }
  __syncthreads();

  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float>
      output_fragments[kOutputFragments];
#pragma unroll
  for (unsigned int fragment = 0U; fragment < kOutputFragments;
       ++fragment) {
    nvcuda::wmma::fill_fragment(output_fragments[fragment], 0.0F);
  }
  float maxima[2] = {-__int_as_float(0x7f800000),
                     -__int_as_float(0x7f800000)};
  float denominators[2] = {0.0F, 0.0F};

  unsigned int current_tile_start =
      warp * kBulkGqaTensorCoreKvTile;
  bool current_valid = current_tile_start < causal_kv_length;
  bulk_gqa_stage_register_kv_tile(
      storage.load.key, key_cache, kv_head, current_tile_start,
      current_valid, warp, lane);
  bulk_gqa_stage_register_kv_tile(
      storage.load.value, value_cache, kv_head, current_tile_start,
      current_valid, warp, lane);

  for (unsigned int iteration = 0U; iteration < iteration_count;
       ++iteration) {
    bulk_gqa_cp_async_wait_group_1();
    __syncthreads();

    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float>
        score_fragment;
    nvcuda::wmma::fill_fragment(score_fragment, 0.0F);
    if (current_valid) {
      const auto* const query_bf16 =
          reinterpret_cast<const __nv_bfloat16*>(storage.load.query);
      const auto* const key_bf16 =
          reinterpret_cast<const __nv_bfloat16*>(storage.load.key) +
          warp * kBulkGqaTensorCoreKvTile * kBulkGqaHeadDimension;
#pragma unroll
      for (unsigned int dimension = 0U;
           dimension < kBulkGqaHeadDimension; dimension += 16U) {
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16,
                               __nv_bfloat16, nvcuda::wmma::row_major>
            query_fragment;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16,
                               __nv_bfloat16, nvcuda::wmma::col_major>
            key_fragment;
        nvcuda::wmma::load_matrix_sync(
            query_fragment, query_bf16 + dimension,
            kBulkGqaHeadDimension);
        nvcuda::wmma::load_matrix_sync(
            key_fragment, key_bf16 + dimension,
            kBulkGqaHeadDimension);
        nvcuda::wmma::mma_sync(score_fragment, query_fragment,
                               key_fragment, score_fragment);
      }
    }

    const unsigned int next_tile_start =
        current_tile_start +
        kBulkGqaRegisterWarpCount * kBulkGqaTensorCoreKvTile;
    const bool next_valid = next_tile_start < causal_kv_length;
    __syncthreads();
    if (iteration + 1U < iteration_count) {
      bulk_gqa_stage_register_kv_tile(
          storage.load.key, key_cache, kv_head, next_tile_start,
          next_valid, warp, lane);
    }
    bulk_gqa_cp_async_wait_group_1();
    __syncthreads();

    if (current_valid) {
      __nv_bfloat16 probability_bits[8];
#pragma unroll
      for (unsigned int reg = 0U; reg < 8U; ++reg) {
        const unsigned int row =
            lane / 4U + 8U * ((reg % 4U) / 2U);
        const unsigned int column =
            2U * (lane % 4U) + 8U * (reg / 4U) + reg % 2U;
        const bool causal = current_tile_start + column <=
                            first_query_token + row;
        score_fragment.x[reg] =
            causal ? score_fragment.x[reg] * kBulkGqaAttentionScale
                   : -__int_as_float(0x7f800000);
      }

#pragma unroll
      for (unsigned int row_group = 0U; row_group < 2U; ++row_group) {
        const unsigned int first = 2U * row_group;
        float local_maximum = fmaxf(
            fmaxf(score_fragment.x[first],
                  score_fragment.x[first + 1U]),
            fmaxf(score_fragment.x[first + 4U],
                  score_fragment.x[first + 5U]));
        local_maximum =
            fmaxf(local_maximum,
                  __shfl_xor_sync(0xffff'ffffU, local_maximum, 1U));
        local_maximum =
            fmaxf(local_maximum,
                  __shfl_xor_sync(0xffff'ffffU, local_maximum, 2U));
        const float previous_maximum = maxima[row_group];
        const float next_maximum =
            fmaxf(previous_maximum, local_maximum);
        const float correction =
            previous_maximum == -__int_as_float(0x7f800000)
                ? 0.0F
                : bulk_gqa_fast_exp(previous_maximum - next_maximum);
        denominators[row_group] *= correction;
#pragma unroll
        for (unsigned int fragment = 0U;
             fragment < kOutputFragments; ++fragment) {
          output_fragments[fragment].x[first] *= correction;
          output_fragments[fragment].x[first + 1U] *= correction;
          output_fragments[fragment].x[first + 4U] *= correction;
          output_fragments[fragment].x[first + 5U] *= correction;
        }
        float local_denominator = 0.0F;
        const unsigned int registers[4] = {
            first, first + 1U, first + 4U, first + 5U};
#pragma unroll
        for (unsigned int item = 0U; item < 4U; ++item) {
          const unsigned int reg = registers[item];
          const float probability =
              bulk_gqa_fast_exp(score_fragment.x[reg] - next_maximum);
          probability_bits[reg] = __float2bfloat16_rn(probability);
          local_denominator += __bfloat162float(probability_bits[reg]);
        }
        local_denominator +=
            __shfl_xor_sync(0xffff'ffffU, local_denominator, 1U);
        local_denominator +=
            __shfl_xor_sync(0xffff'ffffU, local_denominator, 2U);
        denominators[row_group] += local_denominator;
        maxima[row_group] = next_maximum;
      }

      nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16,
                             __nv_bfloat16, nvcuda::wmma::row_major>
          probability_fragment;
#pragma unroll
      for (unsigned int reg = 0U;
           reg < probability_fragment.num_elements; ++reg) {
        probability_fragment.x[reg] = probability_bits[reg];
      }
      const auto* const value_bf16 =
          reinterpret_cast<const __nv_bfloat16*>(storage.load.value) +
          warp * kBulkGqaTensorCoreKvTile * kBulkGqaHeadDimension;
#pragma unroll
      for (unsigned int fragment = 0U; fragment < kOutputFragments;
           ++fragment) {
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16,
                               __nv_bfloat16, nvcuda::wmma::row_major>
            value_fragment;
        nvcuda::wmma::load_matrix_sync(
            value_fragment, value_bf16 + 16U * fragment,
            kBulkGqaHeadDimension);
        nvcuda::wmma::mma_sync(output_fragments[fragment],
                               probability_fragment, value_fragment,
                               output_fragments[fragment]);
      }
    }

    __syncthreads();
    if (iteration + 1U < iteration_count) {
      bulk_gqa_stage_register_kv_tile(
          storage.load.value, value_cache, kv_head, next_tile_start,
          next_valid, warp, lane);
    }
    current_tile_start = next_tile_start;
    current_valid = next_valid;
  }
  bulk_gqa_cp_async_wait_group_0();
  __syncthreads();

  float* const warp_output =
      storage.merge.partial_output + warp * kMatrixElements;
#pragma unroll
  for (unsigned int fragment = 0U; fragment < kOutputFragments;
       ++fragment) {
    nvcuda::wmma::store_matrix_sync(
        warp_output + 16U * fragment, output_fragments[fragment],
        kBulkGqaHeadDimension, nvcuda::wmma::mem_row_major);
  }
  if ((lane & 3U) == 0U) {
    const unsigned int row = lane / 4U;
    storage.merge.maxima[warp * kBulkGqaTensorCoreQueryTile + row] =
        maxima[0];
    storage.merge.maxima[warp * kBulkGqaTensorCoreQueryTile + row + 8U] =
        maxima[1];
    storage.merge.denominators[warp * kBulkGqaTensorCoreQueryTile + row] =
        denominators[0];
    storage.merge.denominators[warp * kBulkGqaTensorCoreQueryTile + row +
                               8U] = denominators[1];
  }
  __syncthreads();

  for (unsigned int index = thread; index < kMatrixElements;
       index += kBulkGqaRegisterThreads) {
    const unsigned int local_query = index / kBulkGqaHeadDimension;
    const unsigned int dimension =
        index - local_query * kBulkGqaHeadDimension;
    float merged_maximum = -__int_as_float(0x7f800000);
#pragma unroll
    for (unsigned int source_warp = 0U;
         source_warp < kBulkGqaRegisterWarpCount; ++source_warp) {
      merged_maximum = fmaxf(
          merged_maximum,
          storage.merge.maxima[source_warp *
                                   kBulkGqaTensorCoreQueryTile +
                               local_query]);
    }
    float merged_denominator = 0.0F;
    float merged_output = 0.0F;
#pragma unroll
    for (unsigned int source_warp = 0U;
         source_warp < kBulkGqaRegisterWarpCount; ++source_warp) {
      const unsigned int row_offset =
          source_warp * kBulkGqaTensorCoreQueryTile + local_query;
      const float source_maximum = storage.merge.maxima[row_offset];
      const float correction =
          source_maximum == -__int_as_float(0x7f800000)
              ? 0.0F
              : bulk_gqa_fast_exp(source_maximum - merged_maximum);
      merged_denominator = fmaf(
          storage.merge.denominators[row_offset], correction,
          merged_denominator);
      merged_output = fmaf(
          storage.merge.partial_output[source_warp * kMatrixElements +
                                       index],
          correction, merged_output);
    }
    const unsigned int token = first_query_token + local_query;
    const std::size_t output_index =
        (static_cast<std::size_t>(token) * kBulkGqaQueryHeads + query_head) *
            kBulkGqaHeadDimension +
        dimension;
    const std::uint16_t rounded_attention =
        encode_bf16_device(merged_output / merged_denominator);
    const float gate_value = decode_bf16_device(gate[output_index]);
    const float gate_exp = bulk_gqa_fast_exp(-fabsf(gate_value));
    const float sigmoid =
        gate_value >= 0.0F ? 1.0F / (1.0F + gate_exp)
                           : gate_exp / (1.0F + gate_exp);
    output[output_index] = encode_bf16_device(
        decode_bf16_device(rounded_attention) * sigmoid);
  }
}

// SM87 Prefill v3.  The flattened query axis follows the
// FlashInfer single-Prefill ownership model: [token, query-in-KV-group] is a
// single 3,072-row axis, blockIdx.x owns 64 rows, and blockIdx.z owns one KV
// head.  Four warps each retain one Q16 online-softmax/output state while the
// CTA loads each K/V32 tile exactly once for all six GQA heads.  P0/C2..C512
// uses ceil(C*6/64) x 1 x 4 CTAs.  P0/C512 retains a separate compile-time
// exact specialization so extending the dataflow does not add predicates to
// its established hot path.
struct alignas(16) BulkGqaGroupSharedStorage {
  alignas(16) std::uint16_t
      query[kBulkGqaGroupQueryTile * kBulkGqaHeadDimension];
  alignas(16) std::uint16_t
      key[kBulkGqaGroupKvTile * kBulkGqaHeadDimension];
  alignas(16) std::uint16_t
      value[kBulkGqaGroupKvTile * kBulkGqaHeadDimension];
};

static_assert(sizeof(BulkGqaGroupSharedStorage) == 64U * 1024U);

template <bool kExactC512>
__device__ __forceinline__ void bulk_gqa_stage_group_kv_tile(
    std::uint16_t* const shared,
    const std::uint16_t* const global,
    const unsigned int kv_head,
    const unsigned int tile_start,
    const unsigned int token_count,
    const unsigned int thread) {
  constexpr unsigned int kVectorsPerHead =
      kBulkGqaHeadDimension * sizeof(std::uint16_t) / sizeof(uint4);
  constexpr unsigned int kVectorsPerTile =
      kBulkGqaGroupKvTile * kVectorsPerHead;
  static_assert(kVectorsPerHead == 32U);
  static_assert(kVectorsPerTile == 1'024U);
#pragma unroll
  for (unsigned int vector = thread; vector < kVectorsPerTile;
       vector += kBulkGqaRegisterThreads) {
    const unsigned int local_position = vector / kVectorsPerHead;
    const unsigned int vector_in_head =
        vector - local_position * kVectorsPerHead;
    if constexpr (kExactC512) {
      const std::size_t global_vector =
          (static_cast<std::size_t>(tile_start + local_position) *
               kBulkGqaKvHeads +
           kv_head) *
              kVectorsPerHead +
          vector_in_head;
      bulk_gqa_cp_async_cg_16(
          shared + vector * (sizeof(uint4) / sizeof(std::uint16_t)),
          reinterpret_cast<const uint4*>(global) + global_vector);
    } else {
      auto* const shared_vector =
          reinterpret_cast<uint4*>(shared) + vector;
      if (tile_start + local_position < token_count) {
        const std::size_t global_vector =
            (static_cast<std::size_t>(tile_start + local_position) *
                 kBulkGqaKvHeads +
             kv_head) *
                kVectorsPerHead +
            vector_in_head;
        bulk_gqa_cp_async_cg_16(
            shared_vector,
            reinterpret_cast<const uint4*>(global) + global_vector);
      } else {
        *shared_vector = make_uint4(0U, 0U, 0U, 0U);
      }
    }
  }
  bulk_gqa_cp_async_commit();
}

template <bool kExactC512>
__global__ __launch_bounds__(kBulkGqaRegisterThreads, 1)
void bulk_causal_gqa_sigmoid_gate_24_4_256_c512_group_q64_kernel(
    const std::uint16_t* const query,
    const std::uint16_t* const key_cache,
    const std::uint16_t* const value_cache,
    const std::uint16_t* const gate,
    std::uint16_t* const output,
    const unsigned int token_count) {
  extern __shared__ __align__(16) unsigned char dynamic_shared[];
  auto& storage =
      *reinterpret_cast<BulkGqaGroupSharedStorage*>(dynamic_shared);

  constexpr unsigned int kQueryVectors =
      kBulkGqaGroupQueryTile * kBulkGqaHeadDimension *
      sizeof(std::uint16_t) / sizeof(uint4);
  constexpr unsigned int kOutputFragments = kBulkGqaHeadDimension / 16U;
  constexpr unsigned int kScoreFragments = kBulkGqaGroupKvTile / 16U;
  static_assert(kQueryVectors == 2'048U);
  static_assert(kOutputFragments == 16U);
  static_assert(kScoreFragments == 2U);

  const unsigned int thread = threadIdx.x;
  const unsigned int warp = thread >> 5U;
  const unsigned int lane = thread & 31U;
  const unsigned int kv_head = blockIdx.z;
  const unsigned int first_packed_query =
      blockIdx.x * kBulkGqaGroupQueryTile;
  const unsigned int warp_packed_query =
      first_packed_query + warp * kBulkGqaTensorCoreQueryTile;
  const unsigned int packed_query_count =
      kExactC512 ? kBulkGqaPackedQueryCount
                 : token_count * kBulkGqaQueriesPerKv;
  const unsigned int remaining_packed_queries =
      packed_query_count - first_packed_query;
  const unsigned int valid_packed_queries =
      kExactC512 || remaining_packed_queries >= kBulkGqaGroupQueryTile
          ? kBulkGqaGroupQueryTile
          : remaining_packed_queries;
  const unsigned int last_packed_query =
      first_packed_query + valid_packed_queries - 1U;
  const unsigned int causal_kv_length =
      last_packed_query / kBulkGqaQueriesPerKv + 1U;
  const unsigned int iteration_count =
      (causal_kv_length + kBulkGqaGroupKvTile - 1U) /
      kBulkGqaGroupKvTile;

  auto* const shared_query_vectors =
      reinterpret_cast<uint4*>(storage.query);
  for (unsigned int vector = thread; vector < kQueryVectors;
       vector += kBulkGqaRegisterThreads) {
    const unsigned int local_packed_query = vector / 32U;
    const unsigned int vector_in_head = vector - local_packed_query * 32U;
    const unsigned int packed_query =
        first_packed_query + local_packed_query;
    if constexpr (kExactC512) {
      const unsigned int query_token =
          packed_query / kBulkGqaQueriesPerKv;
      const unsigned int query_in_group =
          packed_query - query_token * kBulkGqaQueriesPerKv;
      const unsigned int query_head =
          kv_head * kBulkGqaQueriesPerKv + query_in_group;
      const std::size_t global_vector =
          (static_cast<std::size_t>(query_token) * kBulkGqaQueryHeads +
           query_head) *
              32U +
          vector_in_head;
      shared_query_vectors[vector] =
          reinterpret_cast<const uint4*>(query)[global_vector];
    } else if (packed_query < packed_query_count) {
      const unsigned int query_token =
          packed_query / kBulkGqaQueriesPerKv;
      const unsigned int query_in_group =
          packed_query - query_token * kBulkGqaQueriesPerKv;
      const unsigned int query_head =
          kv_head * kBulkGqaQueriesPerKv + query_in_group;
      const std::size_t global_vector =
          (static_cast<std::size_t>(query_token) * kBulkGqaQueryHeads +
           query_head) *
              32U +
          vector_in_head;
      shared_query_vectors[vector] =
          reinterpret_cast<const uint4*>(query)[global_vector];
    } else {
      shared_query_vectors[vector] = make_uint4(0U, 0U, 0U, 0U);
    }
  }
  __syncthreads();

  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float>
      output_fragments[kOutputFragments];
#pragma unroll
  for (unsigned int fragment = 0U; fragment < kOutputFragments;
       ++fragment) {
    nvcuda::wmma::fill_fragment(output_fragments[fragment], 0.0F);
  }
  float maxima[2] = {-__int_as_float(0x7f800000),
                     -__int_as_float(0x7f800000)};
  float denominators[2] = {0.0F, 0.0F};

  unsigned int current_tile_start = 0U;
  bulk_gqa_stage_group_kv_tile<kExactC512>(
      storage.key, key_cache, kv_head, current_tile_start, token_count,
      thread);
  bulk_gqa_stage_group_kv_tile<kExactC512>(
      storage.value, value_cache, kv_head, current_tile_start, token_count,
      thread);

  for (unsigned int iteration = 0U; iteration < iteration_count;
       ++iteration) {
    bulk_gqa_cp_async_wait_group_1();
    __syncthreads();

    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16,
                           float>
        score_fragments[kScoreFragments];
#pragma unroll
    for (unsigned int score = 0U; score < kScoreFragments; ++score) {
      nvcuda::wmma::fill_fragment(score_fragments[score], 0.0F);
    }
    const auto* const query_bf16 =
        reinterpret_cast<const __nv_bfloat16*>(storage.query) +
        warp * kBulkGqaTensorCoreQueryTile * kBulkGqaHeadDimension;
    const auto* const key_bf16 =
        reinterpret_cast<const __nv_bfloat16*>(storage.key);
#pragma unroll
    for (unsigned int dimension = 0U;
         dimension < kBulkGqaHeadDimension; dimension += 16U) {
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16,
                             __nv_bfloat16, nvcuda::wmma::row_major>
          query_fragment;
      nvcuda::wmma::load_matrix_sync(
          query_fragment, query_bf16 + dimension,
          kBulkGqaHeadDimension);
#pragma unroll
      for (unsigned int score = 0U; score < kScoreFragments; ++score) {
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16,
                               __nv_bfloat16, nvcuda::wmma::col_major>
            key_fragment;
        nvcuda::wmma::load_matrix_sync(
            key_fragment,
            key_bf16 + score * kBulkGqaTensorCoreKvTile *
                           kBulkGqaHeadDimension +
                dimension,
            kBulkGqaHeadDimension);
        nvcuda::wmma::mma_sync(score_fragments[score], query_fragment,
                               key_fragment, score_fragments[score]);
      }
    }

    const unsigned int next_tile_start =
        current_tile_start + kBulkGqaGroupKvTile;
    const bool has_next = iteration + 1U < iteration_count;
    __syncthreads();
    if (has_next) {
      bulk_gqa_stage_group_kv_tile<kExactC512>(
          storage.key, key_cache, kv_head, next_tile_start, token_count,
          thread);
      bulk_gqa_cp_async_wait_group_1();
    } else {
      bulk_gqa_cp_async_wait_group_0();
    }
    __syncthreads();

    const auto* const value_bf16 =
        reinterpret_cast<const __nv_bfloat16*>(storage.value);
#pragma unroll
    for (unsigned int score = 0U; score < kScoreFragments; ++score) {
      auto& score_fragment = score_fragments[score];
#pragma unroll
      for (unsigned int reg = 0U; reg < 8U; ++reg) {
        const unsigned int row =
            lane / 4U + 8U * ((reg % 4U) / 2U);
        const unsigned int column =
            2U * (lane % 4U) + 8U * (reg / 4U) + reg % 2U;
        const unsigned int packed_query = warp_packed_query + row;
        const unsigned int query_token =
            packed_query / kBulkGqaQueriesPerKv;
        const unsigned int kv_position =
            current_tile_start + score * kBulkGqaTensorCoreKvTile +
            column;
        if constexpr (kExactC512) {
          score_fragment.x[reg] =
              kv_position <= query_token
                  ? score_fragment.x[reg] * kBulkGqaAttentionScale
                  : -__int_as_float(0x7f800000);
        } else if (packed_query >= packed_query_count) {
          // The incomplete Q64 tail is never stored.  Give its private WMMA
          // rows a finite dummy distribution so -inf - -inf cannot create a
          // NaN that obscures diagnostics for adjacent valid rows.
          score_fragment.x[reg] = 0.0F;
        } else {
          score_fragment.x[reg] =
              kv_position < token_count && kv_position <= query_token
                  ? score_fragment.x[reg] * kBulkGqaAttentionScale
                  : -__int_as_float(0x7f800000);
        }
      }

      __nv_bfloat16 probability_bits[8];
#pragma unroll
      for (unsigned int row_group = 0U; row_group < 2U; ++row_group) {
        const unsigned int first = 2U * row_group;
        float local_maximum = fmaxf(
            fmaxf(score_fragment.x[first],
                  score_fragment.x[first + 1U]),
            fmaxf(score_fragment.x[first + 4U],
                  score_fragment.x[first + 5U]));
        local_maximum =
            fmaxf(local_maximum,
                  __shfl_xor_sync(0xffff'ffffU, local_maximum, 1U));
        local_maximum =
            fmaxf(local_maximum,
                  __shfl_xor_sync(0xffff'ffffU, local_maximum, 2U));
        const float previous_maximum = maxima[row_group];
        const float next_maximum =
            fmaxf(previous_maximum, local_maximum);
        const float correction =
            previous_maximum == -__int_as_float(0x7f800000)
                ? 0.0F
                : bulk_gqa_fast_exp(previous_maximum - next_maximum);
        denominators[row_group] *= correction;
#pragma unroll
        for (unsigned int fragment = 0U;
             fragment < kOutputFragments; ++fragment) {
          output_fragments[fragment].x[first] *= correction;
          output_fragments[fragment].x[first + 1U] *= correction;
          output_fragments[fragment].x[first + 4U] *= correction;
          output_fragments[fragment].x[first + 5U] *= correction;
        }
        float local_denominator = 0.0F;
        const unsigned int registers[4] = {
            first, first + 1U, first + 4U, first + 5U};
#pragma unroll
        for (unsigned int item = 0U; item < 4U; ++item) {
          const unsigned int reg = registers[item];
          const float probability =
              bulk_gqa_fast_exp(score_fragment.x[reg] - next_maximum);
          probability_bits[reg] = __float2bfloat16_rn(probability);
          local_denominator += __bfloat162float(probability_bits[reg]);
        }
        local_denominator +=
            __shfl_xor_sync(0xffff'ffffU, local_denominator, 1U);
        local_denominator +=
            __shfl_xor_sync(0xffff'ffffU, local_denominator, 2U);
        denominators[row_group] += local_denominator;
        maxima[row_group] = next_maximum;
      }

      nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16,
                             __nv_bfloat16, nvcuda::wmma::row_major>
          probability_fragment;
#pragma unroll
      for (unsigned int reg = 0U;
           reg < probability_fragment.num_elements; ++reg) {
        probability_fragment.x[reg] = probability_bits[reg];
      }
#pragma unroll
      for (unsigned int fragment = 0U; fragment < kOutputFragments;
           ++fragment) {
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16,
                               __nv_bfloat16, nvcuda::wmma::row_major>
            value_fragment;
        nvcuda::wmma::load_matrix_sync(
            value_fragment,
            value_bf16 + score * kBulkGqaTensorCoreKvTile *
                             kBulkGqaHeadDimension +
                16U * fragment,
            kBulkGqaHeadDimension);
        nvcuda::wmma::mma_sync(output_fragments[fragment],
                               probability_fragment, value_fragment,
                               output_fragments[fragment]);
      }
    }

    __syncthreads();
    if (has_next) {
      bulk_gqa_stage_group_kv_tile<kExactC512>(
          storage.value, value_cache, kv_head, next_tile_start, token_count,
          thread);
    }
    current_tile_start = next_tile_start;
  }
  bulk_gqa_cp_async_wait_group_0();
  __syncthreads();

#pragma unroll
  for (unsigned int fragment = 0U; fragment < kOutputFragments;
       ++fragment) {
#pragma unroll
    for (unsigned int row_group = 0U; row_group < 2U; ++row_group) {
      const unsigned int first = 2U * row_group;
      const unsigned int packed_query =
          warp_packed_query + lane / 4U + 8U * row_group;
      if (!kExactC512 && packed_query >= packed_query_count) {
        continue;
      }
      const unsigned int query_token =
          packed_query / kBulkGqaQueriesPerKv;
      const unsigned int query_in_group =
          packed_query - query_token * kBulkGqaQueriesPerKv;
      const unsigned int query_head =
          kv_head * kBulkGqaQueriesPerKv + query_in_group;
      const unsigned int first_dimension =
          fragment * 16U + 2U * (lane % 4U);
      const std::size_t low_index =
          (static_cast<std::size_t>(query_token) * kBulkGqaQueryHeads +
           query_head) *
              kBulkGqaHeadDimension +
          first_dimension;
      const std::size_t high_index = low_index + 8U;
      const float inverse_denominator = 1.0F / denominators[row_group];
      const std::uint16_t low_first = bulk_gqa_apply_sigmoid_gate(
          output_fragments[fragment].x[first] * inverse_denominator,
          gate[low_index]);
      const std::uint16_t low_second = bulk_gqa_apply_sigmoid_gate(
          output_fragments[fragment].x[first + 1U] *
              inverse_denominator,
          gate[low_index + 1U]);
      const std::uint16_t high_first = bulk_gqa_apply_sigmoid_gate(
          output_fragments[fragment].x[first + 4U] *
              inverse_denominator,
          gate[high_index]);
      const std::uint16_t high_second = bulk_gqa_apply_sigmoid_gate(
          output_fragments[fragment].x[first + 5U] *
              inverse_denominator,
          gate[high_index + 1U]);
      *reinterpret_cast<std::uint32_t*>(output + low_index) =
          static_cast<std::uint32_t>(low_first) |
          (static_cast<std::uint32_t>(low_second) << 16U);
      *reinterpret_cast<std::uint32_t*>(output + high_index) =
          static_cast<std::uint32_t>(high_first) |
          (static_cast<std::uint32_t>(high_second) << 16U);
    }
  }
}

}  // namespace

int launch_bulk_causal_gqa_sigmoid_gate_24_4_256_c512_register_pipeline_cuda(
    const std::uint16_t* const query,
    const std::uint16_t* const key_cache,
    const std::uint16_t* const value_cache,
    const std::uint16_t* const gate,
    const std::size_t first_position,
    const std::size_t token_count,
    std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  const auto stream = static_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();

  // The final production binary keeps the v2 route as an exact same-ELF
  // comparator.  Production defaults to the KV-head-centric v3 route; setting
  // this narrowly scoped process variable to exactly "1" selects v2 for the
  // mirrored real-P513 gate without changing any other Prefill component.
  static const bool force_v2_baseline = []() noexcept {
    const char* const value =
        std::getenv("Q3X_FULL_ATTENTION_C512_FORCE_V2_BASELINE");
    return value != nullptr && std::strcmp(value, "1") == 0;
  }();
  if (force_v2_baseline && first_position == 0U &&
      token_count == kBulkGqaC512TokenCount) {
    constexpr std::size_t kV2DynamicSharedBytes =
        sizeof(BulkGqaRegisterSharedStorage);
    const cudaError_t attribute_status = cudaFuncSetAttribute(
        bulk_causal_gqa_sigmoid_gate_24_4_256_c512_register_pipeline_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kV2DynamicSharedBytes));
    if (attribute_status != cudaSuccess) {
      return static_cast<int>(attribute_status);
    }
    const dim3 blocks(
        kBulkGqaC512TokenCount / kBulkGqaTensorCoreQueryTile,
        kBulkGqaQueryHeads, 1U);
    bulk_causal_gqa_sigmoid_gate_24_4_256_c512_register_pipeline_kernel
        <<<blocks, kBulkGqaRegisterThreads, kV2DynamicSharedBytes, stream>>>(
            query, key_cache, value_cache, gate, output);
    return static_cast<int>(cudaGetLastError());
  }

  constexpr std::size_t kV3DynamicSharedBytes =
      sizeof(BulkGqaGroupSharedStorage);
  const bool exact_c512 =
      first_position == 0U && token_count == kBulkGqaC512TokenCount;
  const cudaError_t attribute_status =
      exact_c512
          ? cudaFuncSetAttribute(
                bulk_causal_gqa_sigmoid_gate_24_4_256_c512_group_q64_kernel<
                    true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(kV3DynamicSharedBytes))
          : cudaFuncSetAttribute(
                bulk_causal_gqa_sigmoid_gate_24_4_256_c512_group_q64_kernel<
                    false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(kV3DynamicSharedBytes));
  if (attribute_status != cudaSuccess) {
    return static_cast<int>(attribute_status);
  }
  const dim3 blocks(
      exact_c512
          ? kBulkGqaGroupGridX
          : static_cast<unsigned int>(
                (token_count * kBulkGqaQueriesPerKv +
                 kBulkGqaGroupQueryTile - 1U) /
                kBulkGqaGroupQueryTile),
      1U, kBulkGqaKvHeads);
  if (exact_c512) {
    bulk_causal_gqa_sigmoid_gate_24_4_256_c512_group_q64_kernel<true>
        <<<blocks, kBulkGqaRegisterThreads, kV3DynamicSharedBytes, stream>>>(
            query, key_cache, value_cache, gate, output,
            static_cast<unsigned int>(token_count));
  } else {
    bulk_causal_gqa_sigmoid_gate_24_4_256_c512_group_q64_kernel<false>
        <<<blocks, kBulkGqaRegisterThreads, kV3DynamicSharedBytes, stream>>>(
            query, key_cache, value_cache, gate, output,
            static_cast<unsigned int>(token_count));
  }
  return static_cast<int>(cudaGetLastError());
}

}  // namespace q3x::runtime
