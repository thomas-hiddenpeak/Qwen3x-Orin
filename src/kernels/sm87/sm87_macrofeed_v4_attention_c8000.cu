/*
 * Copyright 2026 Qwen3x-Orin contributors.
 *
 * The fixed grouped-query ownership and fragment publication follow the
 * project's established SM87 Attention implementation, whose provenance is
 * recorded in full_attention_c512_register_pipeline.cu.  This is a
 * project-owned fixed-C8000 translation; it contains no vendored FlashInfer
 * include, runtime dispatch, JIT, split-KV merge, or fallback path.
 */
#include "q3x/kernels/sm87_macrofeed_v4_attention_c8000.h"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {
namespace {

constexpr unsigned int kQueryHeads =
    static_cast<unsigned int>(kSm87MacroFeedV4AttentionC8000QueryHeads);
constexpr unsigned int kKvHeads =
    static_cast<unsigned int>(kSm87MacroFeedV4AttentionC8000KvHeads);
constexpr unsigned int kQueriesPerKv =
    static_cast<unsigned int>(kSm87MacroFeedV4AttentionC8000QueriesPerKv);
constexpr unsigned int kHeadDimension =
    static_cast<unsigned int>(kSm87MacroFeedV4AttentionC8000HeadDimension);
constexpr unsigned int kQGateHeadStride =
    static_cast<unsigned int>(kSm87MacroFeedV4AttentionC8000QGateHeadStride);
constexpr unsigned int kScratchRowStride =
    static_cast<unsigned int>(kSm87MacroFeedV4AttentionC8000ScratchRowStride);
constexpr unsigned int kPackedQueryTile =
    static_cast<unsigned int>(kSm87MacroFeedV4AttentionC8000QueryTile);
constexpr unsigned int kKvTile =
    static_cast<unsigned int>(kSm87MacroFeedV4AttentionC8000KvTile);
constexpr unsigned int kThreads =
    static_cast<unsigned int>(kSm87MacroFeedV4AttentionC8000Threads);
constexpr unsigned int kTensorCoreQueryTile = 16U;
constexpr unsigned int kTensorCoreKvTile = 16U;
constexpr unsigned int kBaselinePackedQueryTile = 64U;
constexpr float kAttentionScale = 1.0F / 16.0F;

static_assert(kQueryHeads == 24U && kKvHeads == 4U &&
              kQueriesPerKv == 6U && kHeadDimension == 256U);
static_assert(kPackedQueryTile == 128U && kKvTile == 32U &&
              kThreads == 256U);
static_assert(kThreads ==
              (kPackedQueryTile / kTensorCoreQueryTile) * 32U);

struct alignas(16) AttentionSharedStorage final {
  alignas(16) std::uint16_t query[kPackedQueryTile * kHeadDimension];
  alignas(16) std::uint16_t key[2U][kKvTile * kHeadDimension];
  alignas(16) std::uint16_t value[2U][kKvTile * kHeadDimension];
};

static_assert(sizeof(AttentionSharedStorage) ==
              kSm87MacroFeedV4AttentionC8000DynamicSharedBytes);

__device__ __forceinline__ float decode_bf16(const std::uint16_t value) {
  return __uint_as_float(static_cast<unsigned int>(value) << 16U);
}

__device__ __forceinline__ std::uint16_t encode_bf16_rne(
    const float value) {
  unsigned int bits = __float_as_uint(value);
  const unsigned int magnitude = bits & 0x7fff'ffffU;
  if (magnitude > 0x7f80'0000U) {
    return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
  }
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

__device__ __forceinline__ void cp_async_cg_16(
    void* const shared_destination, const void* const global_source) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const unsigned int shared_address = static_cast<unsigned int>(
      __cvta_generic_to_shared(shared_destination));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :
               : "r"(shared_address), "l"(global_source)
               : "memory");
#else
  *reinterpret_cast<uint4*>(shared_destination) =
      *reinterpret_cast<const uint4*>(global_source);
#endif
}

__device__ __forceinline__ void cp_async_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;" ::: "memory");
#endif
}

__device__ __forceinline__ void cp_async_wait_group_2() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group 2;" ::: "memory");
#endif
}

__device__ __forceinline__ void cp_async_wait_group_1() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group 1;" ::: "memory");
#endif
}

__device__ __forceinline__ void cp_async_wait_group_0() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group 0;" ::: "memory");
#endif
}

__device__ __forceinline__ float fast_exp(const float value) {
#if defined(__CUDA_ARCH__)
  float result = 0.0F;
  asm("ex2.approx.f32 %0, %1;" : "=f"(result)
      : "f"(value * 1.4426950408889634F));
  return result;
#else
  return expf(value);
#endif
}

__device__ __forceinline__ std::uint16_t apply_sigmoid_gate(
    const float attention, const std::uint16_t gate_bits) {
  // This BF16 publication is observable and must remain before sigmoid.
  const std::uint16_t rounded_attention = encode_bf16_rne(attention);
  const float gate_value = decode_bf16(gate_bits);
  const float gate_exp = fast_exp(-fabsf(gate_value));
  const float sigmoid = gate_value >= 0.0F
                            ? 1.0F / (1.0F + gate_exp)
                            : gate_exp / (1.0F + gate_exp);
  return encode_bf16_rne(decode_bf16(rounded_attention) * sigmoid);
}

template <unsigned int kCtaThreads>
__device__ __forceinline__ void stage_kv_tile(
    std::uint16_t* const shared, const std::uint16_t* const global,
    const unsigned int kv_head, const unsigned int tile_start,
    const unsigned int ready_end, const unsigned int thread) {
  constexpr unsigned int kVectorsPerHead =
      kHeadDimension * sizeof(std::uint16_t) / sizeof(uint4);
  constexpr unsigned int kVectorsPerTile = kKvTile * kVectorsPerHead;
  static_assert(kVectorsPerHead == 32U && kVectorsPerTile == 1'024U);
#pragma unroll
  for (unsigned int vector = thread; vector < kVectorsPerTile;
       vector += kCtaThreads) {
    const unsigned int local_position = vector / kVectorsPerHead;
    const unsigned int vector_in_head =
        vector - local_position * kVectorsPerHead;
    auto* const shared_vector = reinterpret_cast<uint4*>(shared) + vector;
    if (tile_start + local_position < ready_end) {
      const unsigned int global_vector =
          ((tile_start + local_position) * kKvHeads + kv_head) *
              kVectorsPerHead +
          vector_in_head;
      cp_async_cg_16(shared_vector,
                     reinterpret_cast<const uint4*>(global) + global_vector);
    } else {
      *shared_vector = make_uint4(0U, 0U, 0U, 0U);
    }
  }
  cp_async_commit();
}

template <bool kExactC8000>
__device__ __forceinline__ void attention_body(
    const std::uint16_t* const q_gate_input,
    std::uint16_t* const q_output,
    const std::uint16_t* const __restrict__ key_cache,
    const std::uint16_t* const __restrict__ value_cache,
    AttentionSharedStorage& storage,
    const unsigned int first_position,
    const unsigned int runtime_token_count) {
  constexpr unsigned int kProductionTokens =
      static_cast<unsigned int>(kSm87MacroFeedV4AttentionC8000Tokens);
  constexpr unsigned int kQueryVectors =
      kPackedQueryTile * kHeadDimension * sizeof(std::uint16_t) /
      sizeof(uint4);
  constexpr unsigned int kOutputFragments = kHeadDimension / 16U;
  constexpr unsigned int kScoreFragments = kKvTile / 16U;
  static_assert(kQueryVectors == 4'096U && kOutputFragments == 16U &&
                kScoreFragments == 2U);

  const unsigned int token_count =
      kExactC8000 ? kProductionTokens : runtime_token_count;
  const unsigned int packed_query_count = token_count * kQueriesPerKv;
  const unsigned int ready_end = first_position + token_count;
  const unsigned int thread = threadIdx.x;
  const unsigned int warp = thread >> 5U;
  const unsigned int lane = thread & 31U;
  const unsigned int kv_head = blockIdx.z;
  const unsigned int first_packed_query = blockIdx.x * kPackedQueryTile;
  const unsigned int warp_packed_query =
      first_packed_query + warp * kTensorCoreQueryTile;
  const unsigned int remaining_packed_queries =
      packed_query_count - first_packed_query;
  const unsigned int valid_packed_queries =
      kExactC8000 || remaining_packed_queries >= kPackedQueryTile
          ? kPackedQueryTile
          : remaining_packed_queries;
  const unsigned int last_packed_query =
      first_packed_query + valid_packed_queries - 1U;
  const unsigned int causal_kv_length =
      first_position + last_packed_query / kQueriesPerKv + 1U;
  const unsigned int iteration_count =
      (causal_kv_length + kKvTile - 1U) / kKvTile;

  // A Q128 CTA is exactly two old Q64 ownership groups.  Each four-warp
  // subgroup retains its own maximum causal iteration so aggregating the CTA
  // cannot introduce extra online-softmax updates into the first subgroup.
  // Every Q row owned by this CTA reaches shared memory before any global Q
  // slot is overwritten.  No other CTA reads these rows, making q_output ==
  // q_gate_input well-defined without a restrict promise.
  auto* const shared_query_vectors =
      reinterpret_cast<uint4*>(storage.query);
  for (unsigned int vector = thread; vector < kQueryVectors;
       vector += kThreads) {
    const unsigned int local_packed_query = vector / 32U;
    const unsigned int vector_in_head = vector - local_packed_query * 32U;
    const unsigned int packed_query =
        first_packed_query + local_packed_query;
    if (kExactC8000 || packed_query < packed_query_count) {
      const unsigned int local_token = packed_query / kQueriesPerKv;
      const unsigned int query_in_group =
          packed_query - local_token * kQueriesPerKv;
      const unsigned int query_head =
          kv_head * kQueriesPerKv + query_in_group;
      const unsigned int physical_element =
          local_token * kScratchRowStride +
          query_head * kQGateHeadStride + vector_in_head * 8U;
      shared_query_vectors[vector] =
          *reinterpret_cast<const uint4*>(q_gate_input + physical_element);
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
  float maxima[2] = {-__int_as_float(0x7f80'0000),
                     -__int_as_float(0x7f80'0000)};
  float denominators[2] = {0.0F, 0.0F};

  stage_kv_tile<kThreads>(storage.key[0U], key_cache, kv_head, 0U,
                          ready_end, thread);
  stage_kv_tile<kThreads>(storage.value[0U], value_cache, kv_head, 0U,
                          ready_end, thread);

  for (unsigned int iteration = 0U; iteration < iteration_count;
       ++iteration) {
    cp_async_wait_group_1();
    __syncthreads();

    const unsigned int baseline_group_end_packed_query =
        first_packed_query + (warp >> 2U) * kBaselinePackedQueryTile +
                kBaselinePackedQueryTile <
            packed_query_count
        ? first_packed_query + (warp >> 2U) * kBaselinePackedQueryTile +
              kBaselinePackedQueryTile
        : packed_query_count;
    const bool baseline_group_iteration_active =
        baseline_group_end_packed_query >
            first_packed_query + (warp >> 2U) * kBaselinePackedQueryTile &&
        iteration * kKvTile <
            first_position +
                (baseline_group_end_packed_query - 1U) / kQueriesPerKv + 1U;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float>
        score_fragments[kScoreFragments];
#pragma unroll
    for (unsigned int score = 0U; score < kScoreFragments; ++score) {
      nvcuda::wmma::fill_fragment(score_fragments[score], 0.0F);
    }
    const auto* const query_bf16 =
        reinterpret_cast<const __nv_bfloat16*>(storage.query) +
        warp * kTensorCoreQueryTile * kHeadDimension;
    const auto* const key_bf16 =
        reinterpret_cast<const __nv_bfloat16*>(storage.key[iteration & 1U]);
    if (baseline_group_iteration_active) {
#pragma unroll
      for (unsigned int dimension = 0U; dimension < kHeadDimension;
           dimension += 16U) {
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16,
                               __nv_bfloat16, nvcuda::wmma::row_major>
            query_fragment;
        nvcuda::wmma::load_matrix_sync(query_fragment,
                                       query_bf16 + dimension,
                                       kHeadDimension);
#pragma unroll
        for (unsigned int score = 0U; score < kScoreFragments; ++score) {
          nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16,
                                 __nv_bfloat16,
                                 nvcuda::wmma::col_major>
              key_fragment;
          nvcuda::wmma::load_matrix_sync(
              key_fragment,
              key_bf16 + score * kTensorCoreKvTile * kHeadDimension +
                  dimension,
              kHeadDimension);
          nvcuda::wmma::mma_sync(score_fragments[score], query_fragment,
                                 key_fragment, score_fragments[score]);
        }
      }
    }

    const bool has_next = iteration + 1U < iteration_count;
    if (has_next) {
      stage_kv_tile<kThreads>(storage.key[(iteration + 1U) & 1U], key_cache,
                              kv_head, (iteration + 1U) * kKvTile, ready_end,
                              thread);
      stage_kv_tile<kThreads>(storage.value[(iteration + 1U) & 1U],
                              value_cache, kv_head,
                              (iteration + 1U) * kKvTile, ready_end, thread);
      cp_async_wait_group_2();
    } else {
      cp_async_wait_group_0();
    }
    __syncthreads();

    const auto* const value_bf16 = reinterpret_cast<const __nv_bfloat16*>(
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
          const unsigned int local_query_token =
              packed_query / kQueriesPerKv;
          const unsigned int query_position =
              first_position + local_query_token;
          const unsigned int kv_position =
              iteration * kKvTile + score * kTensorCoreKvTile + column;
          if (!kExactC8000 && packed_query >= packed_query_count) {
            // Invalid tail rows never publish.  A finite dummy distribution
            // avoids -inf - -inf NaNs from obscuring neighboring diagnostics.
            score_fragment.x[reg] = 0.0F;
          } else {
            score_fragment.x[reg] =
                kv_position < ready_end && kv_position <= query_position
                    ? score_fragment.x[reg] * kAttentionScale
                    : -__int_as_float(0x7f80'0000);
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
          local_maximum = fmaxf(
              local_maximum,
              __shfl_xor_sync(0xffff'ffffU, local_maximum, 1U));
          local_maximum = fmaxf(
              local_maximum,
              __shfl_xor_sync(0xffff'ffffU, local_maximum, 2U));
          const float previous_maximum = maxima[row_group];
          const float next_maximum =
              fmaxf(previous_maximum, local_maximum);
          const float correction =
              previous_maximum == -__int_as_float(0x7f80'0000)
                  ? 0.0F
                  : fast_exp(previous_maximum - next_maximum);
          denominators[row_group] *= correction;
#pragma unroll
          for (unsigned int fragment = 0U; fragment < kOutputFragments;
               ++fragment) {
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
                fast_exp(score_fragment.x[reg] - next_maximum);
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
              value_bf16 + score * kTensorCoreKvTile * kHeadDimension +
                  16U * fragment,
              kHeadDimension);
          nvcuda::wmma::mma_sync(output_fragments[fragment],
                                 probability_fragment, value_fragment,
                                 output_fragments[fragment]);
        }
      }
    }

    // All warps retire PV before the following iteration can recycle this
    // K/V slot as an async destination.
    __syncthreads();
  }
  cp_async_wait_group_0();
  __syncthreads();

#pragma unroll
  for (unsigned int fragment = 0U; fragment < kOutputFragments;
       ++fragment) {
#pragma unroll
    for (unsigned int row_group = 0U; row_group < 2U; ++row_group) {
      const unsigned int first = 2U * row_group;
      const unsigned int packed_query =
          warp_packed_query + lane / 4U + 8U * row_group;
      if (!kExactC8000 && packed_query >= packed_query_count) {
        continue;
      }
      const unsigned int local_token = packed_query / kQueriesPerKv;
      const unsigned int query_in_group =
          packed_query - local_token * kQueriesPerKv;
      const unsigned int query_head =
          kv_head * kQueriesPerKv + query_in_group;
      const unsigned int first_dimension =
          fragment * 16U + 2U * (lane % 4U);
      const unsigned int physical_head =
          local_token * kScratchRowStride +
          query_head * kQGateHeadStride;
      const unsigned int low_index = physical_head + first_dimension;
      const unsigned int high_index = low_index + 8U;
      const unsigned int low_gate =
          low_index + kSm87MacroFeedV4AttentionC8000GateSlotOffset;
      const unsigned int high_gate =
          high_index + kSm87MacroFeedV4AttentionC8000GateSlotOffset;
      const float inverse_denominator = 1.0F / denominators[row_group];
      const std::uint16_t low_first = apply_sigmoid_gate(
          output_fragments[fragment].x[first] * inverse_denominator,
          q_gate_input[low_gate]);
      const std::uint16_t low_second = apply_sigmoid_gate(
          output_fragments[fragment].x[first + 1U] * inverse_denominator,
          q_gate_input[low_gate + 1U]);
      const std::uint16_t high_first = apply_sigmoid_gate(
          output_fragments[fragment].x[first + 4U] * inverse_denominator,
          q_gate_input[high_gate]);
      const std::uint16_t high_second = apply_sigmoid_gate(
          output_fragments[fragment].x[first + 5U] * inverse_denominator,
          q_gate_input[high_gate + 1U]);
      *reinterpret_cast<std::uint32_t*>(q_output + low_index) =
          static_cast<std::uint32_t>(low_first) |
          (static_cast<std::uint32_t>(low_second) << 16U);
      *reinterpret_cast<std::uint32_t*>(q_output + high_index) =
          static_cast<std::uint32_t>(high_first) |
          (static_cast<std::uint32_t>(high_second) << 16U);
    }
  }
}

__global__ __launch_bounds__(kThreads, 1)
void attention_c8000_kernel(
    std::uint16_t* const q_gate_scratch,
    const std::uint16_t* const __restrict__ key_cache,
    const std::uint16_t* const __restrict__ value_cache,
    const unsigned int first_position) {
  extern __shared__ __align__(16) unsigned char dynamic_shared[];
  auto& storage =
      *reinterpret_cast<AttentionSharedStorage*>(dynamic_shared);
  attention_body<true>(q_gate_scratch, q_gate_scratch, key_cache, value_cache,
                       storage, first_position,
                       static_cast<unsigned int>(
                           kSm87MacroFeedV4AttentionC8000Tokens));
}

__global__ __launch_bounds__(kThreads, 1)
void attention_oracle_kernel(
    const std::uint16_t* const q_gate_scratch,
    std::uint16_t* const output_q_gate_scratch,
    const std::uint16_t* const __restrict__ key_cache,
    const std::uint16_t* const __restrict__ value_cache,
    const unsigned int first_position, const unsigned int token_count) {
  extern __shared__ __align__(16) unsigned char dynamic_shared[];
  auto& storage =
      *reinterpret_cast<AttentionSharedStorage*>(dynamic_shared);
  attention_body<false>(q_gate_scratch, output_q_gate_scratch, key_cache,
                        value_cache, storage, first_position, token_count);
}

[[nodiscard]] constexpr bool pointer_aligned(const void* const pointer) {
  return pointer != nullptr &&
         reinterpret_cast<std::uintptr_t>(pointer) %
                 kSm87MacroFeedV4AttentionC8000PointerAlignment ==
             0U;
}

[[nodiscard]] bool structural_arguments_valid(
    const std::uint16_t* const q_gate_input,
    std::uint16_t* const q_output, const std::size_t token_count,
    const std::size_t scratch_row_stride,
    const std::uint16_t* const key_cache,
    const std::uint16_t* const value_cache,
    const std::size_t kv_position_capacity,
    const std::size_t kv_row_stride, const std::size_t first_position,
    const void* const cuda_stream, const bool production_extent) noexcept {
  if ((production_extent &&
       token_count != kSm87MacroFeedV4AttentionC8000Tokens) ||
      (!production_extent && token_count != 1U && token_count != 65U) ||
      scratch_row_stride !=
          kSm87MacroFeedV4AttentionC8000ScratchRowStride ||
      kv_row_stride != kSm87MacroFeedV4AttentionC8000KvRowStride ||
      kv_position_capacity == 0U ||
      kv_position_capacity >
          kSm87MacroFeedV4AttentionC8000MaximumPositions ||
      first_position > kv_position_capacity ||
      token_count > kv_position_capacity - first_position ||
      cuda_stream == nullptr || !pointer_aligned(q_gate_input) ||
      !pointer_aligned(q_output) || !pointer_aligned(key_cache) ||
      !pointer_aligned(value_cache)) {
    return false;
  }
  if (production_extent &&
      (!sm87_macrofeed_v4_attention_c8000_first_position_supported(
           first_position) ||
       kv_position_capacity !=
           kSm87MacroFeedV4AttentionC8000MaximumPositions)) {
    return false;
  }

  const auto scratch_bytes =
      static_cast<std::uint64_t>(token_count) * scratch_row_stride *
      sizeof(std::uint16_t);
  const auto kv_bytes =
      static_cast<std::uint64_t>(kv_position_capacity) * kv_row_stride *
      sizeof(std::uint16_t);
  const auto input = sm87_macrofeed_v4_attention_c8000_byte_range(
      q_gate_input, scratch_bytes);
  const auto output = sm87_macrofeed_v4_attention_c8000_byte_range(
      q_output, scratch_bytes);
  const auto key = sm87_macrofeed_v4_attention_c8000_byte_range(key_cache,
                                                                kv_bytes);
  const auto value = sm87_macrofeed_v4_attention_c8000_byte_range(value_cache,
                                                                  kv_bytes);
  if (!input.valid || !output.valid || !key.valid || !value.valid ||
      !sm87_macrofeed_v4_attention_c8000_ranges_disjoint(key, value) ||
      !sm87_macrofeed_v4_attention_c8000_ranges_disjoint(input, key) ||
      !sm87_macrofeed_v4_attention_c8000_ranges_disjoint(input, value) ||
      !sm87_macrofeed_v4_attention_c8000_ranges_disjoint(output, key) ||
      !sm87_macrofeed_v4_attention_c8000_ranges_disjoint(output, value)) {
    return false;
  }
  return q_gate_input == q_output ||
         sm87_macrofeed_v4_attention_c8000_ranges_disjoint(input, output);
}

[[nodiscard]] cudaError_t validate_fixed_device(
    int* const device, cudaDeviceProp* const properties) noexcept {
  if (device == nullptr || properties == nullptr) {
    return cudaErrorInvalidValue;
  }
  cudaError_t status = cudaGetDevice(device);
  if (status != cudaSuccess) {
    return status;
  }
  status = cudaGetDeviceProperties(properties, *device);
  if (status != cudaSuccess) {
    return status;
  }
  return properties->major == 8 && properties->minor == 7 &&
                 properties->multiProcessorCount ==
                     static_cast<int>(
                         kSm87MacroFeedV4AttentionC8000SmCount) &&
                 properties->sharedMemPerBlockOptin >=
                     kSm87MacroFeedV4AttentionC8000DynamicSharedBytes
             ? cudaSuccess
             : cudaErrorNotSupported;
}

[[nodiscard]] bool device_allocation_range_owned(
    const Sm87MacroFeedV4AttentionC8000ByteRange& range,
    const int device_ordinal) noexcept {
  if (!range.valid || range.begin == 0U || range.end <= range.begin ||
      device_ordinal < 0) {
    return false;
  }
  cudaPointerAttributes attributes{};
  const auto* const pointer = reinterpret_cast<const void*>(range.begin);
  if (cudaPointerGetAttributes(&attributes, pointer) != cudaSuccess ||
      attributes.type != cudaMemoryTypeDevice ||
      attributes.device != device_ordinal) {
    return false;
  }
  CUdeviceptr allocation_base = 0U;
  std::size_t allocation_bytes = 0U;
  if (cuMemGetAddressRange(&allocation_base, &allocation_bytes,
                           static_cast<CUdeviceptr>(range.begin)) !=
          CUDA_SUCCESS ||
      allocation_base == 0U || allocation_bytes == 0U) {
    return false;
  }
  const auto allocation_begin =
      static_cast<std::uintptr_t>(allocation_base);
  if (allocation_begin >
      std::numeric_limits<std::uintptr_t>::max() - allocation_bytes) {
    return false;
  }
  const auto allocation_end = allocation_begin + allocation_bytes;
  return range.begin >= allocation_begin && range.end <= allocation_end;
}

[[nodiscard]] bool device_ranges_owned(
    const std::uint16_t* const q_gate_input,
    std::uint16_t* const q_output, const std::size_t token_count,
    const std::size_t scratch_row_stride,
    const std::uint16_t* const key_cache,
    const std::uint16_t* const value_cache,
    const std::size_t kv_position_capacity,
    const std::size_t kv_row_stride, const int device_ordinal) noexcept {
  const auto scratch_bytes =
      static_cast<std::uint64_t>(token_count) * scratch_row_stride *
      sizeof(std::uint16_t);
  const auto kv_bytes =
      static_cast<std::uint64_t>(kv_position_capacity) * kv_row_stride *
      sizeof(std::uint16_t);
  const auto input = sm87_macrofeed_v4_attention_c8000_byte_range(
      q_gate_input, scratch_bytes);
  const auto output = sm87_macrofeed_v4_attention_c8000_byte_range(
      q_output, scratch_bytes);
  const auto key = sm87_macrofeed_v4_attention_c8000_byte_range(key_cache,
                                                                kv_bytes);
  const auto value = sm87_macrofeed_v4_attention_c8000_byte_range(value_cache,
                                                                  kv_bytes);
  return device_allocation_range_owned(input, device_ordinal) &&
         (q_gate_input == q_output ||
          device_allocation_range_owned(output, device_ordinal)) &&
         device_allocation_range_owned(key, device_ordinal) &&
         device_allocation_range_owned(value, device_ordinal);
}

[[nodiscard]] constexpr bool kernel_resources_equal(
    const Sm87MacroFeedV4AttentionC8000KernelResources& left,
    const Sm87MacroFeedV4AttentionC8000KernelResources& right) noexcept {
  return left.registers_per_thread == right.registers_per_thread &&
         left.static_shared_bytes == right.static_shared_bytes &&
         left.dynamic_shared_bytes == right.dynamic_shared_bytes &&
         left.local_bytes == right.local_bytes &&
         left.maximum_threads_per_block == right.maximum_threads_per_block &&
         left.active_blocks_per_sm == right.active_blocks_per_sm &&
         left.threads_per_block == right.threads_per_block &&
         left.grid_x == right.grid_x && left.grid_y == right.grid_y &&
         left.grid_z == right.grid_z &&
         left.physical_grid_ctas == right.physical_grid_ctas;
}

[[nodiscard]] constexpr bool resource_snapshots_equal(
    const Sm87MacroFeedV4AttentionC8000AdmissionResourceSnapshot& left,
    const Sm87MacroFeedV4AttentionC8000AdmissionResourceSnapshot& right)
    noexcept {
  return left.identity == right.identity &&
         left.device_ordinal == right.device_ordinal &&
         left.compute_major == right.compute_major &&
         left.compute_minor == right.compute_minor &&
         left.sm_count == right.sm_count &&
         left.binary_version == right.binary_version &&
         kernel_resources_equal(left.kernel, right.kernel) &&
         left.kernel_compiled == right.kernel_compiled &&
         left.exact_geometry == right.exact_geometry &&
         left.static_resource_gate_passed ==
             right.static_resource_gate_passed &&
         left.numerical_contract_qualified ==
             right.numerical_contract_qualified &&
         left.production_dispatch_eligible ==
             right.production_dispatch_eligible &&
         left.startup_package_unbound == right.startup_package_unbound &&
         left.execution_capability == right.execution_capability &&
         left.caller_snapshot_grants_production_authority ==
             right.caller_snapshot_grants_production_authority;
}

[[nodiscard]] cudaError_t set_production_dynamic_shared() noexcept {
  return cudaFuncSetAttribute(
      attention_c8000_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(
          kSm87MacroFeedV4AttentionC8000DynamicSharedBytes));
}

[[nodiscard]] cudaError_t set_oracle_dynamic_shared() noexcept {
  return cudaFuncSetAttribute(
      attention_oracle_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(
          kSm87MacroFeedV4AttentionC8000DynamicSharedBytes));
}

}  // namespace

bool sm87_macrofeed_v4_attention_c8000_arguments_valid(
    const Sm87MacroFeedV4AttentionC8000Arguments& arguments) noexcept {
  return sm87_macrofeed_v4_attention_c8000_plan(arguments.first_position,
                                                 arguments.token_count)
             .valid() &&
         structural_arguments_valid(
             arguments.q_gate_scratch, arguments.q_gate_scratch,
             arguments.token_count, arguments.scratch_row_stride,
             arguments.key_cache, arguments.value_cache,
             arguments.kv_position_capacity, arguments.kv_row_stride,
             arguments.first_position, arguments.cuda_stream, true);
}

int query_sm87_macrofeed_v4_attention_c8000_admission_resources_cuda(
    Sm87MacroFeedV4AttentionC8000AdmissionResourceSnapshot* const
        resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = {};
  int device = -1;
  cudaDeviceProp properties{};
  cudaError_t status = validate_fixed_device(&device, &properties);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  status = set_production_dynamic_shared();
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  cudaFuncAttributes attributes{};
  status = cudaFuncGetAttributes(&attributes, attention_c8000_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, attention_c8000_kernel,
      static_cast<int>(kSm87MacroFeedV4AttentionC8000Threads),
      kSm87MacroFeedV4AttentionC8000DynamicSharedBytes);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }

  resources->identity = kSm87MacroFeedV4AttentionC8000Identity;
  resources->device_ordinal = device;
  resources->compute_major = properties.major;
  resources->compute_minor = properties.minor;
  resources->sm_count = properties.multiProcessorCount;
  resources->binary_version = attributes.binaryVersion;
  resources->kernel.registers_per_thread = attributes.numRegs;
  resources->kernel.static_shared_bytes = attributes.sharedSizeBytes;
  resources->kernel.dynamic_shared_bytes =
      kSm87MacroFeedV4AttentionC8000DynamicSharedBytes;
  resources->kernel.local_bytes = attributes.localSizeBytes;
  resources->kernel.maximum_threads_per_block = attributes.maxThreadsPerBlock;
  resources->kernel.active_blocks_per_sm = active_blocks;
  resources->kernel.threads_per_block =
      static_cast<std::int32_t>(kSm87MacroFeedV4AttentionC8000Threads);
  resources->kernel.grid_x =
      static_cast<std::int32_t>(kSm87MacroFeedV4AttentionC8000GridX);
  resources->kernel.grid_y =
      static_cast<std::int32_t>(kSm87MacroFeedV4AttentionC8000GridY);
  resources->kernel.grid_z =
      static_cast<std::int32_t>(kSm87MacroFeedV4AttentionC8000GridZ);
  resources->kernel.physical_grid_ctas =
      static_cast<std::int32_t>(
          kSm87MacroFeedV4AttentionC8000PhysicalCtas);
  resources->kernel_compiled = true;
  resources->exact_geometry = true;
  resources->static_resource_gate_passed = true;
  resources->numerical_contract_qualified = false;
  resources->production_dispatch_eligible = false;
  resources->startup_package_unbound = true;
  resources->execution_capability = false;
  resources->caller_snapshot_grants_production_authority = false;
  resources->static_resource_gate_passed =
      sm87_macrofeed_v4_attention_c8000_admission_resource_gate(*resources);
  return resources->static_resource_gate_passed
             ? static_cast<int>(cudaSuccess)
             : static_cast<int>(cudaErrorLaunchOutOfResources);
}

int launch_sm87_macrofeed_v4_attention_c8000_admission_cuda(
    const Sm87MacroFeedV4AttentionC8000Arguments& arguments,
    const Sm87MacroFeedV4AttentionC8000AdmissionResourceSnapshot& resources,
    Sm87MacroFeedV4AttentionC8000AdmissionLaunchReceipt* const receipt)
    noexcept {
  if (receipt == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *receipt = {};
  if (!sm87_macrofeed_v4_attention_c8000_arguments_valid(arguments)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (!sm87_macrofeed_v4_attention_c8000_admission_resource_gate(resources)) {
    return static_cast<int>(cudaErrorLaunchOutOfResources);
  }
  Sm87MacroFeedV4AttentionC8000AdmissionResourceSnapshot observed{};
  const int query_status =
      query_sm87_macrofeed_v4_attention_c8000_admission_resources_cuda(
          &observed);
  if (query_status != static_cast<int>(cudaSuccess)) {
    return query_status;
  }
  if (resources.device_ordinal != observed.device_ordinal) {
    return static_cast<int>(cudaErrorInvalidDevice);
  }
  if (!resource_snapshots_equal(resources, observed)) {
    return static_cast<int>(cudaErrorLaunchOutOfResources);
  }
  const auto stream = reinterpret_cast<cudaStream_t>(arguments.cuda_stream);
  int stream_device = -1;
  cudaError_t status = cudaStreamGetDevice(stream, &stream_device);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  if (stream_device != observed.device_ordinal) {
    return static_cast<int>(cudaErrorInvalidDevice);
  }
  if (!device_ranges_owned(
          arguments.q_gate_scratch, arguments.q_gate_scratch,
          arguments.token_count, arguments.scratch_row_stride,
          arguments.key_cache, arguments.value_cache,
          arguments.kv_position_capacity, arguments.kv_row_stride,
          observed.device_ordinal)) {
    return static_cast<int>(cudaErrorInvalidDevicePointer);
  }

  const dim3 grid(
      static_cast<unsigned int>(kSm87MacroFeedV4AttentionC8000GridX),
      static_cast<unsigned int>(kSm87MacroFeedV4AttentionC8000GridY),
      static_cast<unsigned int>(kSm87MacroFeedV4AttentionC8000GridZ));
  (void)cudaGetLastError();
  attention_c8000_kernel<<<
      grid, static_cast<unsigned int>(kSm87MacroFeedV4AttentionC8000Threads),
      kSm87MacroFeedV4AttentionC8000DynamicSharedBytes, stream>>>(
      arguments.q_gate_scratch, arguments.key_cache, arguments.value_cache,
      static_cast<unsigned int>(arguments.first_position));
  status = cudaPeekAtLastError();
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }

  receipt->identity = kSm87MacroFeedV4AttentionC8000Identity;
  receipt->device_ordinal = observed.device_ordinal;
  receipt->first_position = arguments.first_position;
  receipt->token_count = arguments.token_count;
  receipt->ready_end = arguments.first_position + arguments.token_count;
  receipt->scratch_row_stride = arguments.scratch_row_stride;
  receipt->kv_position_capacity = arguments.kv_position_capacity;
  receipt->kv_row_stride = arguments.kv_row_stride;
  receipt->split_kv_workspace_bytes = 0U;
  receipt->physical_kernel_launches = 1U;
  receipt->exact_causal = true;
  receipt->online_softmax = true;
  receipt->q64_subgroup_reduction_order = true;
  receipt->q128_fully_staged_before_store = true;
  receipt->q_output_aliases_q_input = true;
  receipt->gate_fused_after_bf16_attention = true;
  receipt->gate_slots_preserved = true;
  receipt->scratch_gap_preserved = true;
  receipt->private_nhd_kv_allocation_origin = true;
  receipt->kv32_ascending_two_stage = true;
  receipt->partition_kv = false;
  receipt->merge_kernel_present = false;
  receipt->temporary_output_present = false;
  receipt->current_device_revalidated = true;
  receipt->caller_snapshot_exact_observed_match = true;
  receipt->device_allocation_ranges_owned = true;
  receipt->caller_stream_non_null = true;
  receipt->live_stream_device_observed = true;
  receipt->stream_owner_verified = false;
  receipt->preprocess_completion_event_bound = false;
  receipt->launch_enqueued = true;
  receipt->completion_observed = false;
  receipt->numerical_contract_qualified = false;
  receipt->production_dispatch_eligible = false;
  receipt->startup_package_unbound = true;
  receipt->execution_capability = false;
  receipt->caller_snapshot_grants_production_authority = false;
  return receipt->valid_enqueue_receipt()
             ? static_cast<int>(cudaSuccess)
             : static_cast<int>(cudaErrorUnknown);
}

int launch_sm87_macrofeed_v4_attention_c8000_oracle_cuda(
    const Sm87MacroFeedV4AttentionC8000OracleArguments& arguments) noexcept {
  if (!structural_arguments_valid(
          arguments.q_gate_scratch, arguments.output_q_gate_scratch,
          arguments.token_count, arguments.scratch_row_stride,
          arguments.key_cache, arguments.value_cache,
          arguments.kv_position_capacity, arguments.kv_row_stride,
          arguments.first_position, arguments.cuda_stream, false)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  int device = -1;
  cudaDeviceProp properties{};
  cudaError_t status = validate_fixed_device(&device, &properties);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  const auto stream = reinterpret_cast<cudaStream_t>(arguments.cuda_stream);
  int stream_device = -1;
  status = cudaStreamGetDevice(stream, &stream_device);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  if (stream_device != device) {
    return static_cast<int>(cudaErrorInvalidDevice);
  }
  if (!device_ranges_owned(
          arguments.q_gate_scratch, arguments.output_q_gate_scratch,
          arguments.token_count, arguments.scratch_row_stride,
          arguments.key_cache, arguments.value_cache,
          arguments.kv_position_capacity, arguments.kv_row_stride, device)) {
    return static_cast<int>(cudaErrorInvalidDevicePointer);
  }
  status = set_oracle_dynamic_shared();
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  const unsigned int grid_x = static_cast<unsigned int>(
      (arguments.token_count * kQueriesPerKv + kPackedQueryTile - 1U) /
      kPackedQueryTile);
  const dim3 grid(grid_x, 1U, kKvHeads);
  (void)cudaGetLastError();
  attention_oracle_kernel<<<
      grid, kThreads, kSm87MacroFeedV4AttentionC8000DynamicSharedBytes,
      stream>>>(arguments.q_gate_scratch, arguments.output_q_gate_scratch,
                arguments.key_cache, arguments.value_cache,
                static_cast<unsigned int>(arguments.first_position),
                static_cast<unsigned int>(arguments.token_count));
  return static_cast<int>(cudaPeekAtLastError());
}

}  // namespace q3x::kernels
