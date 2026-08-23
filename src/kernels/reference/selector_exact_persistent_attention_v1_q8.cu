#include "reference_runner_selector_exact_persistent_attention_v1_internal.h"

#if !defined(Q3X_ENABLE_SELECTOR_EXACT_PERSISTENT_ATTENTION_V1_P40_TESTING)
#error "selector exact persistent attention Q8 is candidate-only"
#endif

#include "q3x/runtime/decode_ops.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::runtime {
namespace {

constexpr unsigned int kQueryHeads = 24U;
constexpr unsigned int kKvHeads = 4U;
constexpr unsigned int kQueriesPerKv = kQueryHeads / kKvHeads;
constexpr unsigned int kHeadDimension = 256U;
constexpr unsigned int kPackedDimension = kHeadDimension / 2U;
constexpr unsigned int kQueryTile = 8U;
constexpr unsigned int kKvTile = 16U;
constexpr unsigned int kThreads = kQueriesPerKv * 32U;
constexpr std::size_t kMaximumSequence =
    kBulkCausalGqaMaximumSequenceLength;
constexpr float kAttentionScale = 1.0F / 16.0F;

static_assert(kQueryHeads % kKvHeads == 0U);
static_assert(kQueriesPerKv == 6U);
static_assert(kPackedDimension * 2U == kHeadDimension);
static_assert(kThreads == 192U);
static_assert(kQueryTile ==
              reference_runner_detail::
                  kSelectorExactPersistentAttentionV1QueryTokens);

[[nodiscard]] bool byte_range_overflows(const void* const pointer,
                                        const std::size_t bytes) noexcept {
  const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
  return bytes > std::numeric_limits<std::uintptr_t>::max() - begin;
}

[[nodiscard]] bool ranges_overlap(const void* const first,
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
  return first_begin < second_begin + second_bytes &&
         second_begin < first_begin + first_bytes;
}

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

// AC-SELECTOR-EXACT-PERSISTENT-ATTENTION-v1. This is deliberately a literal
// Q8 expansion of the incumbent GenericQT2 kernel: one CTA still owns one KV
// head, its six warps still own the same six query heads, and every query
// retains the same scalar FP32 FMA, warp reduction, key-order online softmax,
// FP32 P*V, and BF16->sigmoid-gate->BF16 publication sequence. The changed
// ownership is only that one resident K/V tile feeds eight adjacent queries.
// This translation unit is added solely by the isolated testing candidate.
__global__ __launch_bounds__(kThreads, 1)
void selector_exact_persistent_attention_v1_q8_generic_suffix_kernel(
    const std::uint16_t* const query,
    const std::uint16_t* const key_cache,
    const std::uint16_t* const value_cache,
    const std::uint16_t* const gate,
    const unsigned int first_position,
    const unsigned int token_count,
    std::uint16_t* const output) {
  __shared__ std::uint32_t key_words[kKvTile][kPackedDimension];
  __shared__ std::uint32_t value_words[kKvTile][kPackedDimension];

  constexpr unsigned int kWordsPerLane = kPackedDimension / 32U;
  constexpr unsigned int kValuesPerLane = 2U * kWordsPerLane;
  constexpr unsigned int kWordsPerKvTile = kKvTile * kPackedDimension;
  static_assert(kWordsPerLane == 4U);
  static_assert(kValuesPerLane == 8U);

  const unsigned int thread = threadIdx.x;
  const unsigned int warp = thread >> 5U;
  const unsigned int lane = thread & 31U;
  const unsigned int kv_head = blockIdx.y;
  const unsigned int query_head = kv_head * kQueriesPerKv + warp;
  const unsigned int query_tile_count = token_count / kQueryTile;
  for (unsigned int query_tile = blockIdx.x;
       query_tile < query_tile_count; query_tile += gridDim.x) {
    const unsigned int first_query_token = query_tile * kQueryTile;

    float query_values[kQueryTile][kValuesPerLane];
    float accumulators[kQueryTile][kValuesPerLane];
    float maxima[kQueryTile];
    float denominators[kQueryTile];
#pragma unroll
    for (unsigned int local_query = 0U; local_query < kQueryTile;
         ++local_query) {
      const unsigned int token = first_query_token + local_query;
      const bool valid_query = token < token_count;
      maxima[local_query] = -__int_as_float(0x7f800000);
      denominators[local_query] = 0.0F;
#pragma unroll
      for (unsigned int word_slot = 0U; word_slot < kWordsPerLane;
           ++word_slot) {
        const unsigned int value_slot = 2U * word_slot;
        std::uint32_t packed = 0U;
        if (valid_query) {
          const unsigned int word = lane + 32U * word_slot;
          const std::size_t packed_offset =
              (static_cast<std::size_t>(token) * kQueryHeads + query_head) *
                  kPackedDimension +
              word;
          packed =
              reinterpret_cast<const std::uint32_t*>(query)[packed_offset];
        }
        query_values[local_query][value_slot] =
            decode_bf16_device(static_cast<std::uint16_t>(packed));
        query_values[local_query][value_slot + 1U] = decode_bf16_device(
            static_cast<std::uint16_t>(packed >> 16U));
        accumulators[local_query][value_slot] = 0.0F;
        accumulators[local_query][value_slot + 1U] = 0.0F;
      }
    }

    const unsigned int last_query_token =
        first_query_token + kQueryTile - 1U < token_count
            ? first_query_token + kQueryTile - 1U
            : token_count - 1U;
    const unsigned int causal_kv_length =
        first_position + last_query_token + 1U;
    for (unsigned int kv_tile_start = 0U;
         kv_tile_start < causal_kv_length; kv_tile_start += kKvTile) {
      for (unsigned int packed_index = thread;
           packed_index < 2U * kWordsPerKvTile;
           packed_index += kThreads) {
        const bool is_value = packed_index >= kWordsPerKvTile;
        const unsigned int tile_index =
            is_value ? packed_index - kWordsPerKvTile : packed_index;
        const unsigned int local_position = tile_index / kPackedDimension;
        const unsigned int word =
            tile_index - local_position * kPackedDimension;
        const unsigned int position = kv_tile_start + local_position;
        std::uint32_t packed = 0U;
        if (position < causal_kv_length) {
          const std::size_t cache_offset =
              (static_cast<std::size_t>(position) * kKvHeads + kv_head) *
                  kPackedDimension +
              word;
          packed = is_value
                       ? reinterpret_cast<const std::uint32_t*>(value_cache)
                             [cache_offset]
                       : reinterpret_cast<const std::uint32_t*>(key_cache)
                             [cache_offset];
        }
        if (is_value) {
          value_words[local_position][word] = packed;
        } else {
          key_words[local_position][word] = packed;
        }
      }
      __syncthreads();

      const unsigned int remaining = causal_kv_length - kv_tile_start;
      const unsigned int active_positions =
          remaining < kKvTile ? remaining : kKvTile;
#pragma unroll 1
      for (unsigned int local_position = 0U;
           local_position < active_positions; ++local_position) {
        float key_values[kValuesPerLane];
        float value_values[kValuesPerLane];
#pragma unroll
        for (unsigned int word_slot = 0U; word_slot < kWordsPerLane;
             ++word_slot) {
          const unsigned int word = lane + 32U * word_slot;
          const unsigned int value_slot = 2U * word_slot;
          const std::uint32_t packed_key =
              key_words[local_position][word];
          const std::uint32_t packed_value =
              value_words[local_position][word];
          key_values[value_slot] =
              decode_bf16_device(static_cast<std::uint16_t>(packed_key));
          key_values[value_slot + 1U] = decode_bf16_device(
              static_cast<std::uint16_t>(packed_key >> 16U));
          value_values[value_slot] =
              decode_bf16_device(static_cast<std::uint16_t>(packed_value));
          value_values[value_slot + 1U] = decode_bf16_device(
              static_cast<std::uint16_t>(packed_value >> 16U));
        }
        const unsigned int position = kv_tile_start + local_position;
#pragma unroll
        for (unsigned int local_query = 0U; local_query < kQueryTile;
             ++local_query) {
          const unsigned int token = first_query_token + local_query;
          if (token >= token_count || position > first_position + token) {
            continue;
          }
          float score = 0.0F;
#pragma unroll
          for (unsigned int value_slot = 0U;
               value_slot < kValuesPerLane; ++value_slot) {
            score = fmaf(query_values[local_query][value_slot],
                         key_values[value_slot], score);
          }
#pragma unroll
          for (unsigned int offset = 16U; offset != 0U; offset >>= 1U) {
            const float other =
                __shfl_down_sync(0xffff'ffffU, score, offset);
            if (lane < offset) {
              score += other;
            }
          }
          score = __shfl_sync(0xffff'ffffU, score, 0U) * kAttentionScale;

          if (score > maxima[local_query]) {
            const float correction = expf(maxima[local_query] - score);
            denominators[local_query] =
                denominators[local_query] * correction + 1.0F;
#pragma unroll
            for (unsigned int value_slot = 0U;
                 value_slot < kValuesPerLane; ++value_slot) {
              accumulators[local_query][value_slot] =
                  fmaf(accumulators[local_query][value_slot], correction,
                       value_values[value_slot]);
            }
            maxima[local_query] = score;
          } else {
            const float probability = expf(score - maxima[local_query]);
            denominators[local_query] += probability;
#pragma unroll
            for (unsigned int value_slot = 0U;
                 value_slot < kValuesPerLane; ++value_slot) {
              accumulators[local_query][value_slot] =
                  fmaf(probability, value_values[value_slot],
                       accumulators[local_query][value_slot]);
            }
          }
        }
      }
      __syncthreads();
    }

#pragma unroll
    for (unsigned int local_query = 0U; local_query < kQueryTile;
         ++local_query) {
      const unsigned int token = first_query_token + local_query;
      if (token >= token_count) {
        continue;
      }
#pragma unroll
      for (unsigned int word_slot = 0U; word_slot < kWordsPerLane;
           ++word_slot) {
        const unsigned int word = lane + 32U * word_slot;
        const unsigned int value_slot = 2U * word_slot;
        const std::size_t packed_offset =
            (static_cast<std::size_t>(token) * kQueryHeads + query_head) *
                kPackedDimension +
            word;
        const std::uint32_t packed_gate =
            reinterpret_cast<const std::uint32_t*>(gate)[packed_offset];
        std::uint32_t packed_output = 0U;
#pragma unroll
        for (unsigned int pair_element = 0U; pair_element < 2U;
             ++pair_element) {
          const unsigned int slot = value_slot + pair_element;
          const std::uint16_t rounded_attention = encode_bf16_device(
              accumulators[local_query][slot] /
              denominators[local_query]);
          const float gate_value = decode_bf16_device(
              pair_element == 0U
                  ? static_cast<std::uint16_t>(packed_gate)
                  : static_cast<std::uint16_t>(packed_gate >> 16U));
          const float sigmoid =
              gate_value >= 0.0F
                  ? 1.0F / (1.0F + expf(-gate_value))
                  : expf(gate_value) / (1.0F + expf(gate_value));
          const std::uint16_t gated = encode_bf16_device(
              decode_bf16_device(rounded_attention) * sigmoid);
          packed_output |= static_cast<std::uint32_t>(gated)
                           << (16U * pair_element);
        }
        reinterpret_cast<std::uint32_t*>(output)[packed_offset] =
            packed_output;
      }
    }
  }
}

[[nodiscard]] bool valid_arguments(
    const std::uint16_t* const query,
    const std::uint16_t* const key_cache,
    const std::uint16_t* const value_cache,
    const std::uint16_t* const gate,
    const std::size_t first_position,
    const std::size_t token_count,
    const std::uint16_t* const output) noexcept {
  const bool admitted_suffix =
      (first_position == 257U && token_count == 256U) ||
      (first_position == 1'024U &&
       (token_count == 3'072U || token_count == 7'168U ||
        token_count == 38'976U));
  if (!admitted_suffix || token_count < kQueryTile ||
      token_count % kQueryTile != 0U ||
      first_position > kMaximumSequence - token_count || query == nullptr ||
      key_cache == nullptr || value_cache == nullptr || gate == nullptr ||
      output == nullptr) {
    return false;
  }
  constexpr std::uintptr_t kPackedAlignment = alignof(std::uint32_t);
  if ((reinterpret_cast<std::uintptr_t>(query) % kPackedAlignment) != 0U ||
      (reinterpret_cast<std::uintptr_t>(key_cache) % kPackedAlignment) != 0U ||
      (reinterpret_cast<std::uintptr_t>(value_cache) % kPackedAlignment) !=
          0U ||
      (reinterpret_cast<std::uintptr_t>(gate) % kPackedAlignment) != 0U ||
      (reinterpret_cast<std::uintptr_t>(output) % kPackedAlignment) != 0U) {
    return false;
  }
  if (token_count > std::numeric_limits<std::size_t>::max() /
                        (kQueryHeads * kHeadDimension *
                         sizeof(std::uint16_t)) ||
      first_position + token_count >
          std::numeric_limits<std::size_t>::max() /
              (kKvHeads * kHeadDimension * sizeof(std::uint16_t))) {
    return false;
  }
  const std::size_t query_bytes =
      token_count * kQueryHeads * kHeadDimension * sizeof(std::uint16_t);
  const std::size_t cache_bytes = (first_position + token_count) * kKvHeads *
                                  kHeadDimension * sizeof(std::uint16_t);
  if (byte_range_overflows(query, query_bytes) ||
      byte_range_overflows(key_cache, cache_bytes) ||
      byte_range_overflows(value_cache, cache_bytes) ||
      byte_range_overflows(gate, query_bytes) ||
      byte_range_overflows(output, query_bytes)) {
    return false;
  }
  return !ranges_overlap(query, query_bytes, key_cache, cache_bytes) &&
         !ranges_overlap(query, query_bytes, value_cache, cache_bytes) &&
         !ranges_overlap(query, query_bytes, gate, query_bytes) &&
         !ranges_overlap(query, query_bytes, output, query_bytes) &&
         !ranges_overlap(key_cache, cache_bytes, value_cache, cache_bytes) &&
         !ranges_overlap(key_cache, cache_bytes, gate, query_bytes) &&
         !ranges_overlap(key_cache, cache_bytes, output, query_bytes) &&
         !ranges_overlap(value_cache, cache_bytes, gate, query_bytes) &&
         !ranges_overlap(value_cache, cache_bytes, output, query_bytes) &&
         !ranges_overlap(gate, query_bytes, output, query_bytes);
}

}  // namespace

int reference_runner_detail::
launch_selector_exact_persistent_attention_v1_q8_generic_suffix_cuda(
    const std::uint16_t* const query_suffix,
    const std::uint16_t* const key_cache,
    const std::uint16_t* const value_cache,
    const std::uint16_t* const gate_suffix,
    const std::size_t first_position,
    const std::size_t token_count,
    std::uint16_t* const output_suffix,
    void* const cuda_stream) noexcept {
  if (!valid_arguments(query_suffix, key_cache, value_cache, gate_suffix,
                       first_position, token_count, output_suffix)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const dim3 blocks(
      reference_runner_detail::
          kSelectorExactPersistentAttentionV1PersistentBlocksPerKvHead,
      kKvHeads, 1U);
  const auto stream = static_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  selector_exact_persistent_attention_v1_q8_generic_suffix_kernel
      <<<blocks, kThreads, 0U, stream>>>(
          query_suffix, key_cache, value_cache, gate_suffix,
          static_cast<unsigned int>(first_position),
          static_cast<unsigned int>(token_count), output_suffix);
  return static_cast<int>(cudaGetLastError());
}

int reference_runner_detail::
query_selector_exact_persistent_attention_v1_q8_resources_cuda(
    SelectorExactPersistentAttentionV1Resources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes,
      selector_exact_persistent_attention_v1_q8_generic_suffix_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      selector_exact_persistent_attention_v1_q8_generic_suffix_kernel,
      static_cast<int>(kThreads), 0U);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  resources->registers_per_thread = attributes.numRegs;
  resources->static_shared_bytes = attributes.sharedSizeBytes;
  resources->local_bytes = attributes.localSizeBytes;
  resources->maximum_threads = attributes.maxThreadsPerBlock;
  resources->active_blocks_per_multiprocessor = active_blocks;
  resources->threads_per_block = static_cast<int>(kThreads);
  return static_cast<int>(cudaSuccess);
}

}  // namespace q3x::runtime
