#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

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

static_assert(kBulkGqaQueriesPerKv == 6U);
static_assert(kBulkGqaRegisterThreads == 128U);

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

}  // namespace

int launch_bulk_causal_gqa_sigmoid_gate_24_4_256_c512_register_pipeline_cuda(
    const std::uint16_t* const query,
    const std::uint16_t* const key_cache,
    const std::uint16_t* const value_cache,
    const std::uint16_t* const gate,
    std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  constexpr std::size_t kDynamicSharedBytes =
      sizeof(BulkGqaRegisterSharedStorage);
  const cudaError_t attribute_status = cudaFuncSetAttribute(
      bulk_causal_gqa_sigmoid_gate_24_4_256_c512_register_pipeline_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(kDynamicSharedBytes));
  if (attribute_status != cudaSuccess) {
    return static_cast<int>(attribute_status);
  }

  const dim3 blocks(
      kBulkGqaC512TokenCount / kBulkGqaTensorCoreQueryTile,
      kBulkGqaQueryHeads, 1U);
  const auto stream = static_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  bulk_causal_gqa_sigmoid_gate_24_4_256_c512_register_pipeline_kernel
      <<<blocks, kBulkGqaRegisterThreads, kDynamicSharedBytes, stream>>>(
          query, key_cache, value_cache, gate, output);
  return static_cast<int>(cudaGetLastError());
}

}  // namespace q3x::runtime
