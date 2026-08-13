/*
 * Copyright 2026 Qwen3x-Orin contributors.
 * Portions of the grouped-query Prefill dataflow and fragment-to-output
 * mapping are adapted from FlashInfer, Copyright 2023-2026 FlashInfer
 * community and Copyright 2025-2026 NVIDIA.
 *
 * Licensed under the Apache License, Version 2.0.  This file is a
 * fixed-model, SM87-specific modification and is not part of FlashInfer.
 */
#include "q3x/runtime/decode_ops.h"

#include "q3x/kernels/sm87_bulk_dataflow_v2_attention_l2_cohort.h"

#include "../sm87/sm87_target_aot_attention_launch_internal.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#if defined(Q3X_ENABLE_FLASHINFER_PREFILL_ATTENTION_ADMISSION)
#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/prefill.cuh>
#endif
#include <mma.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>

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
constexpr unsigned int kBulkGqaGroupQ128V4QueryTile =
    static_cast<unsigned int>(
        kBulkCausalGqaGroupQ128V4PackedQueryTile);
constexpr unsigned int kBulkGqaGroupQ128V4Threads =
    static_cast<unsigned int>(kBulkCausalGqaGroupQ128V4Threads);
constexpr unsigned int kBulkGqaGroupGridX =
    kBulkGqaPackedQueryCount / kBulkGqaGroupQueryTile;
constexpr unsigned int kBulkGqaRangeFirstPositionBits =
    kBulkCausalGqaGroupQ64FirstPositionBits;
constexpr unsigned int kBulkGqaRangeFirstPositionMask =
    (1U << kBulkGqaRangeFirstPositionBits) - 1U;

static_assert(kBulkGqaQueriesPerKv == 6U);
static_assert(kBulkGqaRegisterThreads == 128U);
static_assert(kBulkGqaGroupQ128V4QueryTile == 128U);
static_assert(kBulkGqaGroupQ128V4Threads == 256U);
static_assert(kBulkGqaGroupQ128V4Threads / 32U ==
              kBulkGqaGroupQ128V4QueryTile /
                  kBulkGqaTensorCoreQueryTile);
static_assert(kBulkGqaPackedQueryCount == 3'072U);
static_assert(kBulkGqaGroupGridX == 48U);
static_assert(kBulkCausalGqaMaximumSequenceLength <=
              (1U << kBulkGqaRangeFirstPositionBits));
static_assert(kBulkCausalGqaGroupQ64PanelMaximumTokens <
              (1U << (32U - kBulkGqaRangeFirstPositionBits)));

[[nodiscard]] bool byte_range_overflows(const void* const pointer,
                                        const std::size_t bytes) noexcept {
  const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
  return bytes > std::numeric_limits<std::uintptr_t>::max() - begin;
}

[[nodiscard]] bool byte_ranges_overlap(const void* const first,
                                       const std::size_t first_bytes,
                                       const void* const second,
                                       const std::size_t second_bytes) noexcept {
  if (byte_range_overflows(first, first_bytes) ||
      byte_range_overflows(second, second_bytes)) {
    return true;
  }
  const std::uintptr_t first_begin =
      reinterpret_cast<std::uintptr_t>(first);
  const std::uintptr_t second_begin =
      reinterpret_cast<std::uintptr_t>(second);
  const std::uintptr_t first_end = first_begin + first_bytes;
  const std::uintptr_t second_end = second_begin + second_bytes;
  return first_begin < second_end && second_begin < first_end;
}

[[nodiscard]] bool valid_panel_arguments(
    const std::uint16_t* const query,
    const std::uint16_t* const key_cache,
    const std::uint16_t* const value_cache,
    const std::uint16_t* const gate,
    const std::size_t first_position,
    const std::size_t token_count,
    const std::uint16_t* const output,
    const bool geometry_admitted) noexcept {
  if (query == nullptr || key_cache == nullptr || value_cache == nullptr ||
      gate == nullptr || output == nullptr || !geometry_admitted) {
    return false;
  }
  constexpr std::uintptr_t kVectorAlignment = alignof(uint4);
  if ((reinterpret_cast<std::uintptr_t>(query) % kVectorAlignment) != 0U ||
      (reinterpret_cast<std::uintptr_t>(key_cache) % kVectorAlignment) !=
          0U ||
      (reinterpret_cast<std::uintptr_t>(value_cache) % kVectorAlignment) !=
          0U ||
      (reinterpret_cast<std::uintptr_t>(gate) % kVectorAlignment) != 0U ||
      (reinterpret_cast<std::uintptr_t>(output) % kVectorAlignment) != 0U) {
    return false;
  }

  constexpr std::size_t kQueryElementsPerToken =
      kBulkGqaQueryHeads * kBulkGqaHeadDimension;
  constexpr std::size_t kKvElementsPerToken =
      kBulkGqaKvHeads * kBulkGqaHeadDimension;
  const std::size_t panel_bytes =
      token_count * kQueryElementsPerToken * sizeof(std::uint16_t);
  const std::size_t cache_bytes =
      (first_position + token_count) * kKvElementsPerToken *
      sizeof(std::uint16_t);
  if (byte_range_overflows(query, panel_bytes) ||
      byte_range_overflows(key_cache, cache_bytes) ||
      byte_range_overflows(value_cache, cache_bytes) ||
      byte_range_overflows(gate, panel_bytes) ||
      byte_range_overflows(output, panel_bytes)) {
    return false;
  }

  const std::array<std::pair<const void*, std::size_t>, 5U> ranges{{
      {query, panel_bytes},
      {key_cache, cache_bytes},
      {value_cache, cache_bytes},
      {gate, panel_bytes},
      {output, panel_bytes},
  }};
  for (std::size_t first = 0U; first < ranges.size(); ++first) {
    for (std::size_t second = first + 1U; second < ranges.size(); ++second) {
      if (byte_ranges_overlap(ranges[first].first, ranges[first].second,
                              ranges[second].first,
                              ranges[second].second)) {
        return false;
      }
    }
  }
  return true;
}

#if defined(Q3X_ENABLE_FLASHINFER_PREFILL_ATTENTION_ADMISSION)
using FlashInferPrefillParams =
    flashinfer::SinglePrefillParams<__nv_bfloat16, __nv_bfloat16,
                                    __nv_bfloat16>;
#endif

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

#if defined(Q3X_ENABLE_FLASHINFER_PREFILL_ATTENTION_ADMISSION)
__global__ void bulk_gqa_flashinfer_sigmoid_gate_kernel(
    const std::uint16_t* const gate,
    const std::size_t element_count,
    std::uint16_t* const output) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= element_count) {
    return;
  }
  const float attention = decode_bf16_device(output[index]);
  const float gate_value = decode_bf16_device(gate[index]);
  const float gate_exp = exp2f(-fabsf(gate_value) * 1.4426950408889634F);
  const float sigmoid = gate_value >= 0.0F
                            ? 1.0F / (1.0F + gate_exp)
                            : gate_exp / (1.0F + gate_exp);
  output[index] = encode_bf16_device(attention * sigmoid);
}

// Whole-prompt companion.  Capping the resident grid avoids publishing one
// CTA for every 256 elements of a 40K..60K prompt while retaining exactly the
// same per-element instruction and rounding sequence as the panel epilogue.
__global__ void bulk_gqa_flashinfer_sigmoid_gate_whole_prompt_kernel(
    const std::uint16_t* const gate,
    const std::size_t element_count,
    std::uint16_t* const output) {
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < element_count;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const float attention = decode_bf16_device(output[index]);
    const float gate_value = decode_bf16_device(gate[index]);
    const float gate_exp = exp2f(-fabsf(gate_value) * 1.4426950408889634F);
    const float sigmoid = gate_value >= 0.0F
                              ? 1.0F / (1.0F + gate_exp)
                              : gate_exp / (1.0F + gate_exp);
    output[index] = encode_bf16_device(attention * sigmoid);
  }
}

int launch_bulk_gqa_flashinfer_exact_panel_impl(
    const std::uint16_t* const query,
    const std::uint16_t* const key_cache,
    const std::uint16_t* const value_cache,
    const std::uint16_t* const gate,
    const std::size_t first_position,
    const std::size_t token_count,
    std::uint16_t* const output,
    const bool whole_prompt,
    cudaStream_t const stream) noexcept {
  const std::size_t kv_length = first_position + token_count;
  const bool geometry_admitted =
      whole_prompt
          ? can_launch_bulk_causal_gqa_flashinfer_exact_whole_prompt(
                first_position, token_count)
          : can_launch_bulk_causal_gqa_flashinfer_exact_panel(first_position,
                                                               token_count);
  if (!geometry_admitted ||
      kv_length > static_cast<std::size_t>(UINT32_MAX)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  FlashInferPrefillParams params(
      reinterpret_cast<__nv_bfloat16*>(const_cast<std::uint16_t*>(query)),
      reinterpret_cast<__nv_bfloat16*>(
          const_cast<std::uint16_t*>(key_cache)),
      reinterpret_cast<__nv_bfloat16*>(
          const_cast<std::uint16_t*>(value_cache)),
      nullptr, reinterpret_cast<__nv_bfloat16*>(output), nullptr, nullptr,
      kBulkGqaQueryHeads, kBulkGqaKvHeads,
      static_cast<std::uint32_t>(token_count),
      static_cast<std::uint32_t>(kv_length),
      kBulkGqaQueryHeads * kBulkGqaHeadDimension, kBulkGqaHeadDimension,
      kBulkGqaKvHeads * kBulkGqaHeadDimension, kBulkGqaHeadDimension,
      kBulkGqaHeadDimension, -1, 0.0F, kBulkGqaAttentionScale, 1.0F,
      10'000.0F);
  cudaError_t status = cudaErrorUnknown;
  try {
    status = flashinfer::SinglePrefillWithKVCacheDispatched<
        kBulkGqaHeadDimension, kBulkGqaHeadDimension,
        flashinfer::PosEncodingMode::kNone, false,
        flashinfer::MaskMode::kCausal,
        flashinfer::DefaultAttention<false, false, false, false>>(
        params, nullptr, stream);
  } catch (const flashinfer::Error&) {
    return static_cast<int>(cudaErrorNotSupported);
  } catch (...) {
    return static_cast<int>(cudaErrorUnknown);
  }
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  constexpr unsigned int kGateThreads = 256U;
  const std::size_t element_count =
      token_count * kBulkGqaQueryHeads * kBulkGqaHeadDimension;
  const std::size_t required_blocks =
      (element_count + kGateThreads - 1U) / kGateThreads;
  if (whole_prompt) {
    constexpr std::size_t kWholePromptGateMaximumBlocks = 4'096U;
    const auto blocks = static_cast<unsigned int>(
        required_blocks < kWholePromptGateMaximumBlocks
            ? required_blocks
            : kWholePromptGateMaximumBlocks);
    bulk_gqa_flashinfer_sigmoid_gate_whole_prompt_kernel
        <<<blocks, kGateThreads, 0U, stream>>>(gate, element_count, output);
  } else {
    const auto blocks = static_cast<unsigned int>(required_blocks);
    bulk_gqa_flashinfer_sigmoid_gate_kernel<<<blocks, kGateThreads, 0U,
                                              stream>>>(gate, element_count,
                                                        output);
  }
  return static_cast<int>(cudaGetLastError());
}
#endif

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

#if defined(Q3X_ENABLE_SM87_TARGET_AOT_P40_EXECUTOR_V1_ADMISSION) || \
    defined(Q3X_ENABLE_SM87_BULK_DATAFLOW_V2_ATTENTION_L2_COHORT_ADMISSION)
__device__ __forceinline__ void bulk_gqa_cp_async_wait_group_2() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group 2;" ::: "memory");
#endif
}
#endif

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
// and P512/C2..C512 continuations use ceil(C*6/64) x 1 x 4 CTAs.
// Query/Gate/output addresses remain tile-local while K/V and causal positions
// are global. P0/C512 retains a separate compile-time exact specialization so
// extending the dataflow does not add predicates to its established hot path.
template <unsigned int kPackedQueryTile>
struct alignas(16) BulkGqaGroupSharedStorage {
  alignas(16) std::uint16_t
      query[kPackedQueryTile * kBulkGqaHeadDimension];
  alignas(16) std::uint16_t
      key[kBulkGqaGroupKvTile * kBulkGqaHeadDimension];
  alignas(16) std::uint16_t
      value[kBulkGqaGroupKvTile * kBulkGqaHeadDimension];
};

static_assert(sizeof(BulkGqaGroupSharedStorage<kBulkGqaGroupQueryTile>) ==
              64U * 1024U);
static_assert(
    sizeof(BulkGqaGroupSharedStorage<kBulkGqaGroupQ128V4QueryTile>) ==
    kBulkCausalGqaGroupQ128V4DynamicSharedBytes);

#if defined(Q3X_ENABLE_SM87_TARGET_AOT_P40_EXECUTOR_V1_ADMISSION) || \
    defined(Q3X_ENABLE_SM87_BULK_DATAFLOW_V2_ATTENTION_L2_COHORT_ADMISSION)
// The target-AOT body deliberately owns a separate storage type and kernel
// entry so adding the second K/V slot cannot alter the established public
// Q64/Q128 instantiations or their SASS.  Slot i contains one complete K32
// and V32 tile; while PV consumes slot i, cp.async fills slot i^1.
struct alignas(16) TargetAotAttentionQ128Kv32TwoStageSharedStorage {
  alignas(16) std::uint16_t
      query[kBulkGqaGroupQ128V4QueryTile * kBulkGqaHeadDimension];
  alignas(16) std::uint16_t
      key[2U][kBulkGqaGroupKvTile * kBulkGqaHeadDimension];
  alignas(16) std::uint16_t
      value[2U][kBulkGqaGroupKvTile * kBulkGqaHeadDimension];
};

static_assert(
    sizeof(TargetAotAttentionQ128Kv32TwoStageSharedStorage) ==
    kernels::kSm87TargetAotAttentionSharedBytes);
static_assert(
    sizeof(TargetAotAttentionQ128Kv32TwoStageSharedStorage) ==
    128U * 1024U);
#endif

template <bool kExactC512, unsigned int kThreads>
__device__ __forceinline__ void bulk_gqa_stage_group_kv_tile(
    std::uint16_t* const shared,
    const std::uint16_t* const global,
    const unsigned int kv_head,
    const unsigned int tile_start,
    const unsigned int kv_count,
    const unsigned int thread) {
  constexpr unsigned int kVectorsPerHead =
      kBulkGqaHeadDimension * sizeof(std::uint16_t) / sizeof(uint4);
  constexpr unsigned int kVectorsPerTile =
      kBulkGqaGroupKvTile * kVectorsPerHead;
  static_assert(kVectorsPerHead == 32U);
  static_assert(kVectorsPerTile == 1'024U);
#pragma unroll
  for (unsigned int vector = thread; vector < kVectorsPerTile;
       vector += kThreads) {
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
      if (tile_start + local_position < kv_count) {
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

template <bool kExactC512, unsigned int kPackedQueryTile,
          unsigned int kThreads>
__global__ __launch_bounds__(kThreads, 1)
void bulk_causal_gqa_sigmoid_gate_24_4_256_c512_group_q64_kernel(
    const std::uint16_t* const query,
    const std::uint16_t* const key_cache,
    const std::uint16_t* const value_cache,
    const std::uint16_t* const gate,
    std::uint16_t* const output,
    const unsigned int packed_tile_range) {
  extern __shared__ __align__(16) unsigned char dynamic_shared[];
  auto& storage =
      *reinterpret_cast<BulkGqaGroupSharedStorage<kPackedQueryTile>*>(
          dynamic_shared);

  constexpr unsigned int kQueryVectors =
      kPackedQueryTile * kBulkGqaHeadDimension *
      sizeof(std::uint16_t) / sizeof(uint4);
  constexpr unsigned int kOutputFragments = kBulkGqaHeadDimension / 16U;
  constexpr unsigned int kScoreFragments = kBulkGqaGroupKvTile / 16U;
  static_assert(kPackedQueryTile % kBulkGqaTensorCoreQueryTile == 0U);
  static_assert(kThreads ==
                (kPackedQueryTile / kBulkGqaTensorCoreQueryTile) * 32U);
  static_assert(kOutputFragments == 16U);
  static_assert(kScoreFragments == 2U);

  const unsigned int thread = threadIdx.x;
  const unsigned int warp = thread >> 5U;
  const unsigned int lane = thread & 31U;
  const unsigned int kv_head = blockIdx.z;
  const unsigned int first_position =
      kExactC512
          ? 0U
          : packed_tile_range & kBulkGqaRangeFirstPositionMask;
  const unsigned int token_count =
      kExactC512
          ? kBulkGqaC512TokenCount
          : packed_tile_range >> kBulkGqaRangeFirstPositionBits;
  const unsigned int first_packed_query =
      blockIdx.x * kPackedQueryTile;
  const unsigned int warp_packed_query =
      first_packed_query + warp * kBulkGqaTensorCoreQueryTile;
  const unsigned int packed_query_count =
      kExactC512 ? kBulkGqaPackedQueryCount
                 : token_count * kBulkGqaQueriesPerKv;
  const unsigned int remaining_packed_queries =
      packed_query_count - first_packed_query;
  const unsigned int valid_packed_queries =
      kExactC512 || remaining_packed_queries >= kPackedQueryTile
          ? kPackedQueryTile
          : remaining_packed_queries;
  const unsigned int last_packed_query =
      first_packed_query + valid_packed_queries - 1U;
  const unsigned int global_first_position =
      kExactC512 ? 0U : first_position;
  const unsigned int causal_kv_length =
      global_first_position +
      last_packed_query / kBulkGqaQueriesPerKv + 1U;
  const unsigned int global_kv_count =
      kExactC512 ? kBulkGqaC512TokenCount
                 : first_position + token_count;
  const unsigned int iteration_count =
      (causal_kv_length + kBulkGqaGroupKvTile - 1U) /
      kBulkGqaGroupKvTile;
  // Q128-v4 aggregates two established Q64 CTAs only to share each staged
  // K/V32 tile.  Keep the two four-warp subgroups' causal iteration bounds
  // independent so the first subgroup never consumes masked-future values
  // that its original Q64 CTA would not have loaded.  For the production Q64
  // specialization the first disjunct is compile-time true, preserving its
  // hot-path instruction body.
  constexpr unsigned int kBaselinePackedQueryTile =
      kBulkGqaGroupQueryTile;
  const unsigned int baseline_group_first_packed_query =
      first_packed_query +
      (warp >> 2U) * kBaselinePackedQueryTile;
  const unsigned int baseline_group_end_packed_query =
      baseline_group_first_packed_query + kBaselinePackedQueryTile <
              packed_query_count
          ? baseline_group_first_packed_query +
                kBaselinePackedQueryTile
          : packed_query_count;
  const unsigned int baseline_group_iteration_count =
      baseline_group_first_packed_query >= packed_query_count
          ? 0U
          : (global_first_position +
                 (baseline_group_end_packed_query - 1U) /
                     kBulkGqaQueriesPerKv +
                 1U + kBulkGqaGroupKvTile - 1U) /
                kBulkGqaGroupKvTile;

  auto* const shared_query_vectors =
      reinterpret_cast<uint4*>(storage.query);
  for (unsigned int vector = thread; vector < kQueryVectors;
       vector += kThreads) {
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
  bulk_gqa_stage_group_kv_tile<kExactC512, kThreads>(
      storage.key, key_cache, kv_head, current_tile_start, global_kv_count,
      thread);
  bulk_gqa_stage_group_kv_tile<kExactC512, kThreads>(
      storage.value, value_cache, kv_head, current_tile_start, global_kv_count,
      thread);

  for (unsigned int iteration = 0U; iteration < iteration_count;
       ++iteration) {
    bulk_gqa_cp_async_wait_group_1();
    __syncthreads();

    const bool baseline_group_iteration_active =
        kPackedQueryTile == kBaselinePackedQueryTile ||
        iteration < baseline_group_iteration_count;

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
    if (baseline_group_iteration_active) {
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
                                 __nv_bfloat16,
                                 nvcuda::wmma::col_major>
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
    }

    const unsigned int next_tile_start =
        current_tile_start + kBulkGqaGroupKvTile;
    const bool has_next = iteration + 1U < iteration_count;
    __syncthreads();
    if (has_next) {
      bulk_gqa_stage_group_kv_tile<kExactC512, kThreads>(
          storage.key, key_cache, kv_head, next_tile_start, global_kv_count,
          thread);
      bulk_gqa_cp_async_wait_group_1();
    } else {
      bulk_gqa_cp_async_wait_group_0();
    }
    __syncthreads();

    const auto* const value_bf16 =
        reinterpret_cast<const __nv_bfloat16*>(storage.value);
    if (baseline_group_iteration_active) {
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
          const unsigned int local_query_token =
              packed_query / kBulkGqaQueriesPerKv;
          const unsigned int query_position =
              global_first_position + local_query_token;
          const unsigned int kv_position =
              current_tile_start + score * kBulkGqaTensorCoreKvTile +
              column;
          if constexpr (kExactC512) {
            score_fragment.x[reg] =
                kv_position <= query_position
                    ? score_fragment.x[reg] * kBulkGqaAttentionScale
                    : -__int_as_float(0x7f800000);
          } else if (packed_query >= packed_query_count) {
            // The incomplete Q64 tail is never stored.  Give its private WMMA
            // rows a finite dummy distribution so -inf - -inf cannot create a
            // NaN that obscures diagnostics for adjacent valid rows.
            score_fragment.x[reg] = 0.0F;
          } else {
            score_fragment.x[reg] =
                kv_position < global_kv_count &&
                    kv_position <= query_position
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
                               __nv_bfloat16,
                               nvcuda::wmma::row_major>
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
                                 __nv_bfloat16,
                                 nvcuda::wmma::row_major>
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
    }

    __syncthreads();
    if (has_next) {
      bulk_gqa_stage_group_kv_tile<kExactC512, kThreads>(
          storage.value, value_cache, kv_head, next_tile_start,
          global_kv_count,
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

#if defined(Q3X_ENABLE_SM87_TARGET_AOT_P40_EXECUTOR_V1_ADMISSION) || \
    defined(Q3X_ENABLE_SM87_BULK_DATAFLOW_V2_ATTENTION_L2_COHORT_ADMISSION)
// Exact-P40000 target-AOT Attention numerical body.  Both the established
// target-AOT control and the v2 persistent-L2-cohort wrapper call this exact
// function, so changing CTA ownership cannot silently change QK/PV order,
// online-softmax state, or BF16/gate publication.  K/V32 ownership remains a
// true two-slot ping-pong: after QK on slot i, both K and V for slot i^1 are
// issued; wait_group 2 retires only the current V dependency, allowing both
// next-tile copies to overlap the current probability/PV traversal.
__device__ __forceinline__
void sm87_target_aot_attention_q128_kv32_p40_two_stage_body(
    const std::uint16_t* const __restrict__ query,
    const std::uint16_t* const __restrict__ key_cache,
    const std::uint16_t* const __restrict__ value_cache,
    const std::uint16_t* const __restrict__ gate,
    std::uint16_t* const __restrict__ output,
    TargetAotAttentionQ128Kv32TwoStageSharedStorage& storage,
    const unsigned int query_tile, const unsigned int kv_head,
    const bool store_enabled) {

  constexpr unsigned int kPackedQueryTile =
      kBulkGqaGroupQ128V4QueryTile;
  constexpr unsigned int kThreads = kBulkGqaGroupQ128V4Threads;
  constexpr unsigned int kTokenCount = 40'000U;
  constexpr unsigned int kPackedQueryCount =
      kTokenCount * kBulkGqaQueriesPerKv;
  constexpr unsigned int kQueryVectors =
      kPackedQueryTile * kBulkGqaHeadDimension *
      sizeof(std::uint16_t) / sizeof(uint4);
  constexpr unsigned int kOutputFragments =
      kBulkGqaHeadDimension / 16U;
  constexpr unsigned int kScoreFragments =
      kBulkGqaGroupKvTile / 16U;
  constexpr unsigned int kBaselinePackedQueryTile =
      kBulkGqaGroupQueryTile;
  static_assert(kPackedQueryCount % kPackedQueryTile == 0U);
  static_assert(kPackedQueryTile % kBulkGqaTensorCoreQueryTile == 0U);
  static_assert(kThreads ==
                (kPackedQueryTile / kBulkGqaTensorCoreQueryTile) * 32U);
  static_assert(kOutputFragments == 16U);
  static_assert(kScoreFragments == 2U);
  static_assert(kThreads ==
                kernels::kSm87TargetAotAttentionThreads);

  const unsigned int thread = threadIdx.x;
  const unsigned int warp = thread >> 5U;
  const unsigned int lane = thread & 31U;
  const unsigned int first_packed_query =
      query_tile * kPackedQueryTile;
  const unsigned int warp_packed_query =
      first_packed_query + warp * kBulkGqaTensorCoreQueryTile;
  const unsigned int last_packed_query =
      first_packed_query + kPackedQueryTile - 1U;
  const unsigned int causal_kv_length =
      last_packed_query / kBulkGqaQueriesPerKv + 1U;
  const unsigned int iteration_count =
      (causal_kv_length + kBulkGqaGroupKvTile - 1U) /
      kBulkGqaGroupKvTile;
  const unsigned int baseline_group_first_packed_query =
      first_packed_query +
      (warp >> 2U) * kBaselinePackedQueryTile;
  const unsigned int baseline_group_end_packed_query =
      baseline_group_first_packed_query + kBaselinePackedQueryTile;
  const unsigned int baseline_group_iteration_count =
      ((baseline_group_end_packed_query - 1U) /
           kBulkGqaQueriesPerKv +
       1U + kBulkGqaGroupKvTile - 1U) /
      kBulkGqaGroupKvTile;

  auto* const shared_query_vectors =
      reinterpret_cast<uint4*>(storage.query);
  for (unsigned int vector = thread; vector < kQueryVectors;
       vector += kThreads) {
    const unsigned int local_packed_query = vector / 32U;
    const unsigned int vector_in_head = vector - local_packed_query * 32U;
    const unsigned int packed_query =
        first_packed_query + local_packed_query;
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

  bulk_gqa_stage_group_kv_tile<false, kThreads>(
      storage.key[0U], key_cache, kv_head, 0U, kTokenCount, thread);
  bulk_gqa_stage_group_kv_tile<false, kThreads>(
      storage.value[0U], value_cache, kv_head, 0U, kTokenCount, thread);

  for (unsigned int iteration = 0U; iteration < iteration_count;
       ++iteration) {
    bulk_gqa_cp_async_wait_group_1();
    __syncthreads();

    const bool baseline_group_iteration_active =
        iteration < baseline_group_iteration_count;
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
        reinterpret_cast<const __nv_bfloat16*>(
            storage.key[iteration & 1U]);
    if (baseline_group_iteration_active) {
#pragma unroll
      for (unsigned int dimension = 0U;
           dimension < kBulkGqaHeadDimension; dimension += 16U) {
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16,
                               __nv_bfloat16,
                               nvcuda::wmma::row_major>
            query_fragment;
        nvcuda::wmma::load_matrix_sync(
            query_fragment, query_bf16 + dimension,
            kBulkGqaHeadDimension);
#pragma unroll
        for (unsigned int score = 0U; score < kScoreFragments; ++score) {
          nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16,
                                 __nv_bfloat16,
                                 nvcuda::wmma::col_major>
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
    }

    const bool has_next = iteration + 1U < iteration_count;
    if (has_next) {
      bulk_gqa_stage_group_kv_tile<false, kThreads>(
          storage.key[(iteration + 1U) & 1U], key_cache, kv_head,
          (iteration + 1U) * kBulkGqaGroupKvTile, kTokenCount, thread);
      bulk_gqa_stage_group_kv_tile<false, kThreads>(
          storage.value[(iteration + 1U) & 1U], value_cache, kv_head,
          (iteration + 1U) * kBulkGqaGroupKvTile, kTokenCount, thread);
      bulk_gqa_cp_async_wait_group_2();
    } else {
      bulk_gqa_cp_async_wait_group_0();
    }
    __syncthreads();

    const auto* const value_bf16 =
        reinterpret_cast<const __nv_bfloat16*>(
            storage.value[iteration & 1U]);
    if (baseline_group_iteration_active) {
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
          const unsigned int query_position =
              packed_query / kBulkGqaQueriesPerKv;
          const unsigned int kv_position =
              iteration * kBulkGqaGroupKvTile +
              score * kBulkGqaTensorCoreKvTile +
              column;
          score_fragment.x[reg] =
              kv_position <= query_position
                  ? score_fragment.x[reg] * kBulkGqaAttentionScale
                  : -__int_as_float(0x7f800000);
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
                               __nv_bfloat16,
                               nvcuda::wmma::row_major>
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
                                 __nv_bfloat16,
                                 nvcuda::wmma::row_major>
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
    }

    // All warps must retire current-slot PV before the following iteration
    // can recycle this slot as the next async destination.
    __syncthreads();
  }
  bulk_gqa_cp_async_wait_group_0();
  __syncthreads();

#pragma unroll
  for (unsigned int fragment = 0U; fragment < kOutputFragments;
       ++fragment) {
#pragma unroll
    for (unsigned int row_group = 0U; row_group < 2U; ++row_group) {
      if (!store_enabled) {
        continue;
      }
      const unsigned int first = 2U * row_group;
      const unsigned int packed_query =
          warp_packed_query + lane / 4U + 8U * row_group;
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
          output_fragments[fragment].x[first + 1U] * inverse_denominator,
          gate[low_index + 1U]);
      const std::uint16_t high_first = bulk_gqa_apply_sigmoid_gate(
          output_fragments[fragment].x[first + 4U] * inverse_denominator,
          gate[high_index]);
      const std::uint16_t high_second = bulk_gqa_apply_sigmoid_gate(
          output_fragments[fragment].x[first + 5U] * inverse_denominator,
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

#if defined(Q3X_ENABLE_SM87_TARGET_AOT_P40_EXECUTOR_V1_ADMISSION)
__global__ __launch_bounds__(kBulkGqaGroupQ128V4Threads, 1)
void sm87_target_aot_attention_q128_kv32_p40_two_stage_kernel(
    const std::uint16_t* const __restrict__ query,
    const std::uint16_t* const __restrict__ key_cache,
    const std::uint16_t* const __restrict__ value_cache,
    const std::uint16_t* const __restrict__ gate,
    std::uint16_t* const __restrict__ output) {
  extern __shared__ __align__(16) unsigned char dynamic_shared[];
  auto& storage =
      *reinterpret_cast<
          TargetAotAttentionQ128Kv32TwoStageSharedStorage*>(
          dynamic_shared);
  sm87_target_aot_attention_q128_kv32_p40_two_stage_body(
      query, key_cache, value_cache, gate, output, storage, blockIdx.x,
      blockIdx.z, true);
}
#endif

#if defined(Q3X_ENABLE_SM87_BULK_DATAFLOW_V2_ATTENTION_L2_COHORT_ADMISSION)
// One launch binds one KV head and exactly 16 persistent CTA lanes.  No
// cooperative launch, cross-CTA barrier, counter, or lock exists.  The host
// issues four stream-ordered launches so a resident wave cannot mix KV heads.
__global__ __launch_bounds__(kBulkGqaGroupQ128V4Threads, 1)
void sm87_bulk_v2_attention_q128_kv32_p40_l2_cohort_kernel(
    const std::uint16_t* const __restrict__ query,
    const std::uint16_t* const __restrict__ key_cache,
    const std::uint16_t* const __restrict__ value_cache,
    const std::uint16_t* const __restrict__ gate,
    std::uint16_t* const __restrict__ output,
    const unsigned int kv_head) {
  extern __shared__ __align__(16) unsigned char dynamic_shared[];
  auto& storage =
      *reinterpret_cast<
          TargetAotAttentionQ128Kv32TwoStageSharedStorage*>(
          dynamic_shared);
  const unsigned int persistent_lane = blockIdx.x;
#pragma unroll 1
  for (unsigned int epoch = 0U;
       epoch < kernels::kSm87BulkV2AttentionSnakeEpochs; ++epoch) {
    const auto work = kernels::sm87_bulk_v2_attention_work_item(
        kv_head, persistent_lane, epoch);
    sm87_target_aot_attention_q128_kv32_p40_two_stage_body(
        query, key_cache, value_cache, gate, output, storage,
        static_cast<unsigned int>(work.query_tile), kv_head,
        work.store_enabled);
  }
}
#endif
#endif

[[nodiscard]] int launch_bulk_gqa_group_q64_v3_fixed_impl(
    const std::uint16_t* const query,
    const std::uint16_t* const key_cache,
    const std::uint16_t* const value_cache,
    const std::uint16_t* const gate,
    const std::size_t first_position,
    const std::size_t token_count,
    std::uint16_t* const output,
    const cudaStream_t stream) noexcept {
  constexpr std::size_t kV3DynamicSharedBytes =
      sizeof(BulkGqaGroupSharedStorage<kBulkGqaGroupQueryTile>);
  const bool exact_c512 =
      first_position == 0U && token_count == kBulkGqaC512TokenCount;
  const cudaError_t attribute_status =
      exact_c512
          ? cudaFuncSetAttribute(
                bulk_causal_gqa_sigmoid_gate_24_4_256_c512_group_q64_kernel<
                    true, kBulkGqaGroupQueryTile,
                    kBulkGqaRegisterThreads>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(kV3DynamicSharedBytes))
          : cudaFuncSetAttribute(
                bulk_causal_gqa_sigmoid_gate_24_4_256_c512_group_q64_kernel<
                    false, kBulkGqaGroupQueryTile,
                    kBulkGqaRegisterThreads>,
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
    bulk_causal_gqa_sigmoid_gate_24_4_256_c512_group_q64_kernel<
        true, kBulkGqaGroupQueryTile, kBulkGqaRegisterThreads>
        <<<blocks, kBulkGqaRegisterThreads, kV3DynamicSharedBytes, stream>>>(
            query, key_cache, value_cache, gate, output,
            static_cast<unsigned int>(token_count));
  } else {
    const unsigned int packed_tile_range =
        static_cast<unsigned int>(first_position) |
        (static_cast<unsigned int>(token_count)
         << kBulkGqaRangeFirstPositionBits);
    bulk_causal_gqa_sigmoid_gate_24_4_256_c512_group_q64_kernel<
        false, kBulkGqaGroupQueryTile, kBulkGqaRegisterThreads>
        <<<blocks, kBulkGqaRegisterThreads, kV3DynamicSharedBytes, stream>>>(
            query, key_cache, value_cache, gate, output,
            packed_tile_range);
  }
  return static_cast<int>(cudaGetLastError());
}

[[nodiscard]] int launch_bulk_gqa_group_q128_v4_fixed_impl(
    const std::uint16_t* const query,
    const std::uint16_t* const key_cache,
    const std::uint16_t* const value_cache,
    const std::uint16_t* const gate,
    const std::size_t first_position,
    const std::size_t token_count,
    std::uint16_t* const output,
    const cudaStream_t stream) noexcept {
  constexpr std::size_t kV4DynamicSharedBytes =
      sizeof(BulkGqaGroupSharedStorage<kBulkGqaGroupQ128V4QueryTile>);
  static_assert(kV4DynamicSharedBytes ==
                kBulkCausalGqaGroupQ128V4DynamicSharedBytes);
  const auto kernel =
      bulk_causal_gqa_sigmoid_gate_24_4_256_c512_group_q64_kernel<
          false, kBulkGqaGroupQ128V4QueryTile,
          kBulkGqaGroupQ128V4Threads>;
  const cudaError_t attribute_status = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(kV4DynamicSharedBytes));
  if (attribute_status != cudaSuccess) {
    return static_cast<int>(attribute_status);
  }
  const dim3 blocks(
      static_cast<unsigned int>(
          bulk_causal_gqa_group_q128_v4_grid_x(token_count)),
      1U, kBulkGqaKvHeads);
  const unsigned int packed_tile_range =
      static_cast<unsigned int>(first_position) |
      (static_cast<unsigned int>(token_count)
       << kBulkGqaRangeFirstPositionBits);
  kernel<<<blocks, kBulkGqaGroupQ128V4Threads,
           kV4DynamicSharedBytes, stream>>>(
      query, key_cache, value_cache, gate, output, packed_tile_range);
  return static_cast<int>(cudaGetLastError());
}

}  // namespace

bool has_bulk_causal_gqa_flashinfer_exact_panel_cuda() noexcept {
#if defined(Q3X_ENABLE_FLASHINFER_PREFILL_ATTENTION_ADMISSION)
  return true;
#else
  return false;
#endif
}

int launch_bulk_causal_gqa_sigmoid_gate_24_4_256_flashinfer_exact_panel_fixed_cuda(
    const std::uint16_t* const query,
    const std::uint16_t* const key_cache,
    const std::uint16_t* const value_cache,
    const std::uint16_t* const gate,
    const std::size_t first_position,
    const std::size_t token_count,
    std::uint16_t* const output,
    void* const cuda_stream) noexcept {
#if defined(Q3X_ENABLE_FLASHINFER_PREFILL_ATTENTION_ADMISSION)
  if (!valid_panel_arguments(
          query, key_cache, value_cache, gate, first_position, token_count,
          output,
          can_launch_bulk_causal_gqa_flashinfer_exact_panel(first_position,
                                                            token_count))) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const auto stream = static_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  return launch_bulk_gqa_flashinfer_exact_panel_impl(
      query, key_cache, value_cache, gate, first_position, token_count,
      output, false, stream);
#else
  (void)query;
  (void)key_cache;
  (void)value_cache;
  (void)gate;
  (void)first_position;
  (void)token_count;
  (void)output;
  (void)cuda_stream;
  return static_cast<int>(cudaErrorNotSupported);
#endif
}

int launch_bulk_causal_gqa_sigmoid_gate_24_4_256_flashinfer_exact_whole_prompt_fixed_cuda(
    const std::uint16_t* const query,
    const std::uint16_t* const key_cache,
    const std::uint16_t* const value_cache,
    const std::uint16_t* const gate,
    const std::size_t first_position,
    const std::size_t token_count,
    std::uint16_t* const output,
    void* const cuda_stream) noexcept {
#if defined(Q3X_ENABLE_FLASHINFER_PREFILL_ATTENTION_ADMISSION)
  if (!valid_panel_arguments(
          query, key_cache, value_cache, gate, first_position, token_count,
          output,
          can_launch_bulk_causal_gqa_flashinfer_exact_whole_prompt(
              first_position, token_count))) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const auto stream = static_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  return launch_bulk_gqa_flashinfer_exact_panel_impl(
      query, key_cache, value_cache, gate, first_position, token_count,
      output, true, stream);
#else
  (void)query;
  (void)key_cache;
  (void)value_cache;
  (void)gate;
  (void)first_position;
  (void)token_count;
  (void)output;
  (void)cuda_stream;
  return static_cast<int>(cudaErrorNotSupported);
#endif
}

int launch_bulk_causal_gqa_sigmoid_gate_24_4_256_c512_group_q64_v3_fixed_cuda(
    const std::uint16_t* const query,
    const std::uint16_t* const key_cache,
    const std::uint16_t* const value_cache,
    const std::uint16_t* const gate,
    const std::size_t first_position,
    const std::size_t token_count,
    std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  if (query == nullptr || key_cache == nullptr || value_cache == nullptr ||
      gate == nullptr || output == nullptr ||
      !use_bulk_causal_gqa_group_q64_prefill(first_position, token_count)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const auto stream = static_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  return launch_bulk_gqa_group_q64_v3_fixed_impl(
      query, key_cache, value_cache, gate, first_position, token_count,
      output, stream);
}

int launch_bulk_causal_gqa_sigmoid_gate_24_4_256_group_q64_panel_fixed_cuda(
    const std::uint16_t* const query,
    const std::uint16_t* const key_cache,
    const std::uint16_t* const value_cache,
    const std::uint16_t* const gate,
    const std::size_t first_position,
    const std::size_t token_count,
    std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  if (!valid_panel_arguments(
          query, key_cache, value_cache, gate, first_position, token_count,
          output,
          can_launch_bulk_causal_gqa_group_q64_panel(first_position,
                                                     token_count))) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const auto stream = static_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  return launch_bulk_gqa_group_q64_v3_fixed_impl(
      query, key_cache, value_cache, gate, first_position, token_count,
      output, stream);
}

int launch_bulk_causal_gqa_sigmoid_gate_24_4_256_group_q128_v4_panel_fixed_cuda(
    const std::uint16_t* const query,
    const std::uint16_t* const key_cache,
    const std::uint16_t* const value_cache,
    const std::uint16_t* const gate,
    const std::size_t first_position,
    const std::size_t token_count,
    std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  if (!can_launch_bulk_causal_gqa_group_q128_v4_panel(first_position,
                                                       token_count) ||
      !valid_panel_arguments(query, key_cache, value_cache, gate,
                             first_position, token_count, output, true)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const auto stream = static_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  return launch_bulk_gqa_group_q128_v4_fixed_impl(
      query, key_cache, value_cache, gate, first_position, token_count,
      output, stream);
}

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

#if defined(Q3X_ENABLE_FLASHINFER_PREFILL_ATTENTION_ADMISSION)
  static const bool use_flashinfer_direct = []() noexcept {
    const char* const value =
        std::getenv("Q3X_FULL_ATTENTION_FLASHINFER_DIRECT");
    return value != nullptr && std::strcmp(value, "1") == 0;
  }();
  if (use_flashinfer_direct) {
    return launch_bulk_gqa_flashinfer_exact_panel_impl(
        query, key_cache, value_cache, gate, first_position, token_count,
        output, false, stream);
  }
#endif

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

  return launch_bulk_gqa_group_q64_v3_fixed_impl(
      query, key_cache, value_cache, gate, first_position, token_count,
      output, stream);
}

namespace sm87_target_aot_attention_execution_detail {

#if defined(Q3X_ENABLE_SM87_TARGET_AOT_P40_EXECUTOR_V1_ADMISSION)
namespace {

constexpr unsigned int kTargetP40GridX =
    static_cast<unsigned int>(
        kTargetP40TokenCount * kBulkGqaQueriesPerKv /
        kBulkGqaGroupQ128V4QueryTile);
static_assert(kTargetP40GridX == 1'875U);
static_assert(kTargetP40TokenCount * kBulkGqaQueriesPerKv %
                      kBulkGqaGroupQ128V4QueryTile ==
                  0U);

[[nodiscard]] cudaError_t validate_target_p40_device(
    const std::int32_t device_ordinal,
    cudaDeviceProp* const properties) noexcept {
  if (device_ordinal < 0 || properties == nullptr) {
    return cudaErrorInvalidValue;
  }
  int current_device = -1;
  cudaError_t status = cudaGetDevice(&current_device);
  if (status != cudaSuccess) {
    return status;
  }
  if (current_device != device_ordinal) {
    return cudaErrorInvalidDevice;
  }
  status = cudaGetDeviceProperties(properties, current_device);
  if (status != cudaSuccess) {
    return status;
  }
  return properties->major == 8 && properties->minor == 7 &&
                 properties->multiProcessorCount == 16 &&
                 properties->warpSize == 32 &&
                 properties->maxThreadsPerBlock >=
                     static_cast<int>(
                         kernels::kSm87TargetAotAttentionThreads) &&
                 properties->sharedMemPerBlockOptin >=
                     kernels::kSm87TargetAotAttentionSharedBytes
             ? cudaSuccess
             : cudaErrorNotSupported;
}

[[nodiscard]] bool target_p40_exact_device_range(
    const void* const pointer, const std::size_t bytes,
    const std::int32_t device_ordinal) noexcept {
  const auto range = target_p40_byte_range(pointer, bytes);
  if (!range.valid) {
    return false;
  }
  const void* const endpoints[2U] = {
      pointer, reinterpret_cast<const void*>(range.end - 1U)};
  for (const void* const endpoint : endpoints) {
    cudaPointerAttributes attributes{};
    if (cudaPointerGetAttributes(&attributes, endpoint) != cudaSuccess ||
        attributes.type != cudaMemoryTypeDevice ||
        attributes.device != device_ordinal) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] cudaError_t validate_target_p40_device_ranges(
    const TargetP40Arguments& arguments) noexcept {
  const std::array<std::pair<const void*, std::size_t>, 5U> ranges{{
      {arguments.processed_query, kTargetP40QueryBytes},
      {arguments.processed_key, kTargetP40KvBytes},
      {arguments.processed_value, kTargetP40KvBytes},
      {arguments.processed_gate, kTargetP40QueryBytes},
      {arguments.gated_output, kTargetP40QueryBytes},
  }};
  for (const auto& range : ranges) {
    if (!target_p40_exact_device_range(
            range.first, range.second, arguments.device_ordinal)) {
      return cudaErrorInvalidDevicePointer;
    }
  }
  return cudaSuccess;
}

[[nodiscard]] cudaError_t set_target_p40_dynamic_shared() noexcept {
  return cudaFuncSetAttribute(
      sm87_target_aot_attention_q128_kv32_p40_two_stage_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(kernels::kSm87TargetAotAttentionSharedBytes));
}

}  // namespace

int query_q128_kv32_p40_two_stage_resources(
    const std::int32_t device_ordinal,
    TargetP40Resources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = {};

  cudaDeviceProp properties{};
  cudaError_t status =
      validate_target_p40_device(device_ordinal, &properties);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  status = set_target_p40_dynamic_shared();
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }

  cudaFuncAttributes attributes{};
  status = cudaFuncGetAttributes(
      &attributes,
      sm87_target_aot_attention_q128_kv32_p40_two_stage_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      sm87_target_aot_attention_q128_kv32_p40_two_stage_kernel,
      static_cast<int>(kernels::kSm87TargetAotAttentionThreads),
      kernels::kSm87TargetAotAttentionSharedBytes);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }

  resources->token_count = kTargetP40TokenCount;
  resources->query_rows = kernels::kSm87TargetAotAttentionQueryRows;
  resources->kv_tokens = kernels::kSm87TargetAotAttentionKvTokens;
  resources->pipeline_stages =
      kernels::kSm87TargetAotAttentionPipelineStages;
  resources->threads = kernels::kSm87TargetAotAttentionThreads;
  resources->warps = kernels::kSm87TargetAotAttentionWarps;
  resources->binary_version = attributes.binaryVersion;
  resources->registers_per_thread = attributes.numRegs;
  resources->static_shared_bytes = attributes.sharedSizeBytes;
  resources->dynamic_shared_bytes =
      kernels::kSm87TargetAotAttentionSharedBytes;
  resources->local_bytes = attributes.localSizeBytes;
  resources->maximum_threads_per_block = attributes.maxThreadsPerBlock;
  resources->active_blocks_per_sm = active_blocks;
  resources->device_ordinal = device_ordinal;
  resources->device_sm_count = properties.multiProcessorCount;
  resources->device_optin_shared_bytes =
      properties.sharedMemPerBlockOptin;
  resources->kernel_compiled = true;
  resources->exact_p40000_only = true;
  resources->cp_async_kv = true;
  resources->kv_ping_pong = true;
  // Observation is not admission.  Numerical and whole-executor evidence
  // must qualify this body before any production dispatcher can name it.
  resources->static_resources_qualified = false;
  resources->numerical_contract_qualified = false;
  resources->production_dispatch_eligible = false;
  return target_p40_resources_structurally_valid(*resources)
             ? static_cast<int>(cudaSuccess)
             : static_cast<int>(cudaErrorNotSupported);
}

int launch_q128_kv32_p40_two_stage(
    const TargetP40Arguments& arguments) noexcept {
  if (!target_p40_arguments_structurally_valid(arguments)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  TargetP40Resources resources{};
  const int resource_status = query_q128_kv32_p40_two_stage_resources(
      arguments.device_ordinal, &resources);
  if (resource_status != static_cast<int>(cudaSuccess)) {
    return resource_status;
  }
  const cudaError_t range_status =
      validate_target_p40_device_ranges(arguments);
  if (range_status != cudaSuccess) {
    return static_cast<int>(range_status);
  }

  const auto stream = reinterpret_cast<cudaStream_t>(arguments.cuda_stream);
  const dim3 blocks(kTargetP40GridX, 1U, kBulkGqaKvHeads);
  (void)cudaGetLastError();
  sm87_target_aot_attention_q128_kv32_p40_two_stage_kernel<<<
      blocks, kernels::kSm87TargetAotAttentionThreads,
      kernels::kSm87TargetAotAttentionSharedBytes, stream>>>(
      arguments.processed_query, arguments.processed_key,
      arguments.processed_value, arguments.processed_gate,
      arguments.gated_output);
  return static_cast<int>(cudaPeekAtLastError());
}
#else
int query_q128_kv32_p40_two_stage_resources(
    const std::int32_t device_ordinal,
    TargetP40Resources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  (void)device_ordinal;
  *resources = {};
  return static_cast<int>(cudaErrorNotSupported);
}

int launch_q128_kv32_p40_two_stage(
    const TargetP40Arguments& arguments) noexcept {
  if (!target_p40_arguments_structurally_valid(arguments)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  return static_cast<int>(cudaErrorNotSupported);
}
#endif

}  // namespace sm87_target_aot_attention_execution_detail

}  // namespace q3x::runtime

namespace q3x::kernels {

#if defined(Q3X_ENABLE_SM87_BULK_DATAFLOW_V2_ATTENTION_L2_COHORT_ADMISSION)
namespace {

[[nodiscard]] cudaError_t validate_bulk_v2_attention_device(
    const std::int32_t device_ordinal,
    cudaDeviceProp* const properties) noexcept {
  if (device_ordinal < 0 || properties == nullptr) {
    return cudaErrorInvalidValue;
  }
  int current_device = -1;
  cudaError_t status = cudaGetDevice(&current_device);
  if (status != cudaSuccess) {
    return status;
  }
  if (current_device != device_ordinal) {
    return cudaErrorInvalidDevice;
  }
  status = cudaGetDeviceProperties(properties, current_device);
  if (status != cudaSuccess) {
    return status;
  }
  return properties->major == 8 && properties->minor == 7 &&
                 properties->multiProcessorCount ==
                     kSm87BulkV2AttentionRequiredSmCount &&
                 properties->warpSize == 32 &&
                 properties->maxThreadsPerBlock >=
                     static_cast<int>(kSm87BulkV2AttentionThreads) &&
                 properties->sharedMemPerBlockOptin >=
                     kSm87BulkV2AttentionDynamicSharedBytes
             ? cudaSuccess
             : cudaErrorNotSupported;
}

[[nodiscard]] bool bulk_v2_attention_device_range_valid(
    const Sm87BulkV2AttentionByteRange& range,
    const std::int32_t device_ordinal) noexcept {
  if (!range.valid) {
    return false;
  }
  const void* const endpoints[2U] = {
      reinterpret_cast<const void*>(range.begin),
      reinterpret_cast<const void*>(range.end - 1U)};
  for (const void* const endpoint : endpoints) {
    cudaPointerAttributes attributes{};
    if (cudaPointerGetAttributes(&attributes, endpoint) != cudaSuccess ||
        attributes.type != cudaMemoryTypeDevice ||
        attributes.device != device_ordinal) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] cudaError_t validate_bulk_v2_attention_device_ranges(
    const Sm87BulkV2AttentionArguments& arguments) noexcept {
  for (const auto& range :
       sm87_bulk_v2_attention_argument_ranges(arguments)) {
    if (!bulk_v2_attention_device_range_valid(range,
                                               arguments.device_ordinal)) {
      return cudaErrorInvalidDevicePointer;
    }
  }
  return cudaSuccess;
}

[[nodiscard]] cudaError_t set_bulk_v2_attention_dynamic_shared() noexcept {
  return cudaFuncSetAttribute(
      runtime::sm87_bulk_v2_attention_q128_kv32_p40_l2_cohort_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(kSm87BulkV2AttentionDynamicSharedBytes));
}

}  // namespace
#endif

int query_sm87_bulk_dataflow_v2_attention_l2_cohort_resources_cuda(
    const std::int32_t device_ordinal,
    Sm87BulkV2AttentionResources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = {};
#if defined(Q3X_ENABLE_SM87_BULK_DATAFLOW_V2_ATTENTION_L2_COHORT_ADMISSION)
  cudaDeviceProp properties{};
  cudaError_t status =
      validate_bulk_v2_attention_device(device_ordinal, &properties);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  status = set_bulk_v2_attention_dynamic_shared();
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }

  cudaFuncAttributes attributes{};
  status = cudaFuncGetAttributes(
      &attributes,
      runtime::sm87_bulk_v2_attention_q128_kv32_p40_l2_cohort_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      runtime::sm87_bulk_v2_attention_q128_kv32_p40_l2_cohort_kernel,
      static_cast<int>(kSm87BulkV2AttentionThreads),
      kSm87BulkV2AttentionDynamicSharedBytes);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }

  resources->binary_version = attributes.binaryVersion;
  resources->registers_per_thread = attributes.numRegs;
  resources->static_shared_bytes = attributes.sharedSizeBytes;
  resources->dynamic_shared_bytes =
      kSm87BulkV2AttentionDynamicSharedBytes;
  resources->local_bytes = attributes.localSizeBytes;
  resources->maximum_threads_per_block = attributes.maxThreadsPerBlock;
  resources->active_blocks_per_sm = active_blocks;
  resources->device_sm_count = properties.multiProcessorCount;
  resources->device_optin_shared_bytes = properties.sharedMemPerBlockOptin;
  resources->threads_per_block =
      static_cast<int>(kSm87BulkV2AttentionThreads);
  resources->physical_grid_ctas_per_launch =
      static_cast<int>(kSm87BulkV2AttentionPersistentLanes);
  resources->physical_launches =
      static_cast<int>(kSm87BulkV2AttentionKernelLaunches);
  resources->query_tiles_per_kv_head =
      kSm87BulkV2AttentionQueryTilesPerKvHead;
  resources->snake_epochs = kSm87BulkV2AttentionSnakeEpochs;
  resources->store_disabled_bodies =
      kSm87BulkV2AttentionStoreDisabledBodies;
  resources->kernel_compiled = true;
  resources->exact_p40000_only = true;
  resources->same_kv_head_per_launch = true;
  resources->mapping_bijective =
      sm87_bulk_v2_attention_mapping_is_bijective();
  resources->no_cooperative_launch = true;
  resources->no_cross_cta_barrier_or_lock = true;
  resources->persistent_cta_residency_capacity =
      active_blocks * properties.multiProcessorCount >=
      static_cast<int>(kSm87BulkV2AttentionPersistentLanes);
  resources->resource_gate_passed = true;
  // This first cell has mapping/static-resource authority only.  It cannot
  // manufacture numerical qualification or a production selector.
  resources->numerical_contract_qualified = false;
  resources->production_dispatch_eligible = false;
  return sm87_bulk_v2_attention_resources_valid(*resources)
             ? static_cast<int>(cudaSuccess)
             : static_cast<int>(cudaErrorNotSupported);
#else
  (void)device_ordinal;
  return static_cast<int>(cudaErrorNotSupported);
#endif
}

int launch_sm87_bulk_dataflow_v2_attention_l2_cohort_cuda(
    const Sm87BulkV2AttentionArguments& arguments) noexcept {
  if (!sm87_bulk_v2_attention_arguments_valid(arguments)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
#if defined(Q3X_ENABLE_SM87_BULK_DATAFLOW_V2_ATTENTION_L2_COHORT_ADMISSION)
  Sm87BulkV2AttentionResources resources{};
  const int resource_status =
      query_sm87_bulk_dataflow_v2_attention_l2_cohort_resources_cuda(
          arguments.device_ordinal, &resources);
  if (resource_status != static_cast<int>(cudaSuccess)) {
    return resource_status;
  }
  const cudaError_t range_status =
      validate_bulk_v2_attention_device_ranges(arguments);
  if (range_status != cudaSuccess) {
    return static_cast<int>(range_status);
  }

  const auto stream = reinterpret_cast<cudaStream_t>(arguments.cuda_stream);
  const dim3 blocks(
      static_cast<unsigned int>(kSm87BulkV2AttentionPersistentLanes), 1U,
      1U);
  (void)cudaGetLastError();
  for (unsigned int kv_head = 0U;
       kv_head < kSm87BulkV2AttentionKvHeads; ++kv_head) {
    runtime::sm87_bulk_v2_attention_q128_kv32_p40_l2_cohort_kernel<<<
        blocks, kSm87BulkV2AttentionThreads,
        kSm87BulkV2AttentionDynamicSharedBytes, stream>>>(
        arguments.processed_query, arguments.processed_key,
        arguments.processed_value, arguments.processed_gate,
        arguments.gated_output, kv_head);
    const cudaError_t launch_status = cudaPeekAtLastError();
    if (launch_status != cudaSuccess) {
      return static_cast<int>(launch_status);
    }
  }
  return static_cast<int>(cudaSuccess);
#else
  return static_cast<int>(cudaErrorNotSupported);
#endif
}

}  // namespace q3x::kernels
