#include "gdn_prefill_chunk_o_bv64_sm87.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace q3x::runtime::gdn_prefill_chunk_o_bv64_detail {
namespace {

constexpr unsigned int kChunk = 64U;
constexpr unsigned int kKeyDimension = 128U;
constexpr unsigned int kValueDimension = 128U;
constexpr unsigned int kQkHeads = 16U;
constexpr unsigned int kValueHeads = 48U;
constexpr unsigned int kHeadGroup = 3U;
constexpr unsigned int kChunkThreads = 128U;
constexpr unsigned int kChunkWarps = kChunkThreads / 32U;
constexpr unsigned int kNormThreads = 256U;
constexpr unsigned int kNormRowsPerCta = kNormThreads / 32U;
constexpr unsigned int kBv = 64U;
constexpr unsigned int kBk = 64U;
constexpr unsigned int kN8Panels = kBv / 8U;
constexpr unsigned int kK16PerPanel = kBk / 16U;
constexpr unsigned int kKeyPanels = kKeyDimension / kBk;
constexpr unsigned int kSharedTileElements = kChunk * kBk;
constexpr unsigned int kSharedTileBytes =
    kSharedTileElements * sizeof(std::uint16_t);
constexpr unsigned int kSharedBytes = 3U * kSharedTileBytes;
constexpr unsigned int kQOffset = 0U;
constexpr unsigned int kKOffset = kSharedTileBytes;
constexpr unsigned int kHOffset = 2U * kSharedTileBytes;
constexpr unsigned int kAOffset = 0U;
constexpr unsigned int kVOffset = kSharedTileBytes;
constexpr unsigned int kGammaOffset = 2U * kSharedTileBytes;
constexpr unsigned int kRowsPerToken = kValueHeads;
constexpr unsigned int kValuesPerToken =
    kValueHeads * kValueDimension;
constexpr unsigned int kMaximumTokens = 512U;

static_assert(kQkHeads * kHeadGroup == kValueHeads);
static_assert(kChunkWarps == 4U);
static_assert(kNormRowsPerCta == 8U);
static_assert(kSharedTileBytes == 8'192U);
static_assert(kSharedBytes == 24'576U);

struct M16K16Fragment final {
  std::uint32_t x0;
  std::uint32_t x1;
  std::uint32_t x2;
  std::uint32_t x3;
};

struct K16N8Fragment final {
  std::uint32_t x0;
  std::uint32_t x1;
};

struct M16N8Accumulator final {
  float x0;
  float x1;
  float x2;
  float x3;
};

[[nodiscard]] __device__ __forceinline__ unsigned int shared_index(
    const unsigned int row, const unsigned int column) {
  // Triton's winning SM87 artifact uses vec=8, perPhase=1, maxPhase=8.
  // In BF16 units this XORs the row phase into the 16-byte column vector.
  return row * kBk + (column ^ ((row & 7U) * 8U));
}

__device__ __forceinline__ void cp_async_cg_16(
    void* const destination, const void* const source) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const unsigned int shared_address =
      static_cast<unsigned int>(__cvta_generic_to_shared(destination));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :
               : "r"(shared_address), "l"(source)
               : "memory");
#else
  (void)destination;
  (void)source;
#endif
}

__device__ __forceinline__ void cp_async_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;" ::: "memory");
#endif
}

__device__ __forceinline__ void cp_async_wait_all() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group 0;" ::: "memory");
#endif
}

__device__ __forceinline__ void load_m16k16(
    M16K16Fragment& fragment, const std::uint16_t* const tile,
    const unsigned int m_panel, const unsigned int k16,
    const unsigned int lane) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
  const unsigned int quadrant = lane / 8U;
  const unsigned int row =
      m_panel * 16U + (lane % 8U) + (quadrant & 1U) * 8U;
  const unsigned int column = k16 * 16U + (quadrant >> 1U) * 8U;
  const auto* const source = tile + shared_index(row, column);
  const unsigned int shared_address =
      static_cast<unsigned int>(__cvta_generic_to_shared(source));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(fragment.x0), "=r"(fragment.x1),
        "=r"(fragment.x2), "=r"(fragment.x3)
      : "r"(shared_address)
      : "memory");
#else
  (void)fragment;
  (void)tile;
  (void)m_panel;
  (void)k16;
  (void)lane;
#endif
}

__device__ __forceinline__ void load_k16n8_transposed(
    K16N8Fragment& fragment, const std::uint16_t* const tile,
    const unsigned int n_panel, const unsigned int k16,
    const unsigned int lane) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
  // tile is canonical [N,K]. Its column-major view is the B operand [K,N].
  const unsigned int row = n_panel * 8U + (lane & 7U);
  const unsigned int column =
      k16 * 16U + ((lane >> 3U) & 1U) * 8U;
  const auto* const source = tile + shared_index(row, column);
  const unsigned int shared_address =
      static_cast<unsigned int>(__cvta_generic_to_shared(source));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
      "{%0, %1}, [%2];"
      : "=r"(fragment.x0), "=r"(fragment.x1)
      : "r"(shared_address)
      : "memory");
#else
  (void)fragment;
  (void)tile;
  (void)n_panel;
  (void)k16;
  (void)lane;
#endif
}

__device__ __forceinline__ void mma_m16n8k16(
    M16N8Accumulator& accumulator,
    const M16K16Fragment& activation,
    const K16N8Fragment& weight) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(accumulator.x0), "+f"(accumulator.x1),
        "+f"(accumulator.x2), "+f"(accumulator.x3)
      : "r"(activation.x0), "r"(activation.x1),
        "r"(activation.x2), "r"(activation.x3),
        "r"(weight.x0), "r"(weight.x1));
#else
  (void)accumulator;
  (void)activation;
  (void)weight;
#endif
}

[[nodiscard]] __device__ __forceinline__ M16N8Accumulator zero_accumulator() {
  return M16N8Accumulator{0.0F, 0.0F, 0.0F, 0.0F};
}

[[nodiscard]] __device__ __forceinline__ float decode_bf16(
    const std::uint16_t value) {
  return __uint_as_float(static_cast<unsigned int>(value) << 16U);
}

[[nodiscard]] __device__ __forceinline__ std::uint16_t encode_bf16(
    const float value) {
  const unsigned int bits = __float_as_uint(value);
  if ((bits & 0x7fff'ffffU) > 0x7f80'0000U) {
    return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
  }
  return __bfloat16_as_ushort(__float2bfloat16_rn(value));
}

[[nodiscard]] __device__ __forceinline__ std::uint32_t pack_bf16_pair(
    const std::uint16_t low, const std::uint16_t high) {
  return static_cast<std::uint32_t>(low) |
         (static_cast<std::uint32_t>(high) << 16U);
}

__device__ __forceinline__ void store_fp32_accumulator(
    float* const destination, const M16N8Accumulator& accumulator,
    const unsigned int m_panel, const unsigned int n_panel,
    const unsigned int lane) {
  const unsigned int row0 = m_panel * 16U + lane / 4U;
  const unsigned int row1 = row0 + 8U;
  const unsigned int column = n_panel * 8U + 2U * (lane & 3U);
  *reinterpret_cast<float2*>(destination + row0 * kBv + column) =
      make_float2(accumulator.x0, accumulator.x1);
  *reinterpret_cast<float2*>(destination + row1 * kBv + column) =
      make_float2(accumulator.x2, accumulator.x3);
}

__device__ __forceinline__ void load_fp32_accumulator(
    M16N8Accumulator& accumulator, const float* const source,
    const unsigned int m_panel, const unsigned int n_panel,
    const unsigned int lane) {
  const unsigned int row0 = m_panel * 16U + lane / 4U;
  const unsigned int row1 = row0 + 8U;
  const unsigned int column = n_panel * 8U + 2U * (lane & 3U);
  const float2 low =
      *reinterpret_cast<const float2*>(source + row0 * kBv + column);
  const float2 high =
      *reinterpret_cast<const float2*>(source + row1 * kBv + column);
  accumulator = M16N8Accumulator{low.x, low.y, high.x, high.y};
}

[[nodiscard]] __device__ __forceinline__ std::uint16_t gated_score(
    const float raw, const unsigned int query, const unsigned int source,
    const float* const gamma) {
  if (source > query) {
    return 0U;
  }
  return encode_bf16(raw * expf(gamma[query] - gamma[source]));
}

__device__ __forceinline__ void store_gated_score_accumulator(
    std::uint16_t* const destination,
    const M16N8Accumulator& accumulator,
    const unsigned int m_panel, const unsigned int n_panel,
    const unsigned int lane, const float* const gamma) {
  const unsigned int query0 = m_panel * 16U + lane / 4U;
  const unsigned int query1 = query0 + 8U;
  const unsigned int source0 = n_panel * 8U + 2U * (lane & 3U);
  const unsigned int source1 = source0 + 1U;
  *reinterpret_cast<std::uint32_t*>(
      destination + shared_index(query0, source0)) =
      pack_bf16_pair(gated_score(accumulator.x0, query0, source0, gamma),
                     gated_score(accumulator.x1, query0, source1, gamma));
  *reinterpret_cast<std::uint32_t*>(
      destination + shared_index(query1, source0)) =
      pack_bf16_pair(gated_score(accumulator.x2, query1, source0, gamma),
                     gated_score(accumulator.x3, query1, source1, gamma));
}

__device__ __forceinline__ void store_raw_output_accumulator(
    std::uint16_t* const raw_output,
    const M16N8Accumulator& accumulator,
    const unsigned int token_base, const unsigned int value_head,
    const unsigned int value_base, const unsigned int m_panel,
    const unsigned int n_panel, const unsigned int lane) {
  const unsigned int query0 = m_panel * 16U + lane / 4U;
  const unsigned int query1 = query0 + 8U;
  const unsigned int value =
      value_base + n_panel * 8U + 2U * (lane & 3U);
  *reinterpret_cast<std::uint32_t*>(
      raw_output +
      static_cast<std::size_t>(token_base + query0) * kValuesPerToken +
      value_head * kValueDimension + value) =
      pack_bf16_pair(encode_bf16(accumulator.x0),
                     encode_bf16(accumulator.x1));
  *reinterpret_cast<std::uint32_t*>(
      raw_output +
      static_cast<std::size_t>(token_base + query1) * kValuesPerToken +
      value_head * kValueDimension + value) =
      pack_bf16_pair(encode_bf16(accumulator.x2),
                     encode_bf16(accumulator.x3));
}

__global__ __launch_bounds__(kChunkThreads, 3)
void chunk_o_bv64_kernel(
    const std::uint16_t* const compact_q,
    const std::uint16_t* const compact_k,
    const std::uint16_t* const boundary_state,
    const std::uint16_t* const v_new,
    const float* const cumulative_gate,
    std::uint16_t* const raw_output) {
  __shared__ __align__(16) unsigned char shared_storage[kSharedBytes];
  auto* const shared_q = reinterpret_cast<std::uint16_t*>(
      shared_storage + kQOffset);
  auto* const shared_k = reinterpret_cast<std::uint16_t*>(
      shared_storage + kKOffset);
  auto* const shared_h = reinterpret_cast<std::uint16_t*>(
      shared_storage + kHOffset);

  const unsigned int bv_index = blockIdx.x;
  const unsigned int chunk_index = blockIdx.y;
  const unsigned int value_head = blockIdx.z;
  const unsigned int qk_head = value_head / kHeadGroup;
  const unsigned int value_base = bv_index * kBv;
  const unsigned int token_base = chunk_index * kChunk;
  const std::size_t compact_matrix =
      static_cast<std::size_t>(chunk_index) * kQkHeads + qk_head;
  const std::size_t value_matrix =
      static_cast<std::size_t>(chunk_index) * kValueHeads + value_head;
  const auto* const matrix_q =
      compact_q + compact_matrix * kChunk * kKeyDimension;
  const auto* const matrix_k =
      compact_k + compact_matrix * kChunk * kKeyDimension;
  const auto* const matrix_h =
      boundary_state + value_matrix * kValueDimension * kKeyDimension;
  const auto* const matrix_v =
      v_new + value_matrix * kChunk * kValueDimension;

  const unsigned int warp = threadIdx.x / 32U;
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int query_panel = warp;

  M16N8Accumulator state[kN8Panels];
  M16N8Accumulator score[kN8Panels];
#pragma unroll
  for (unsigned int panel = 0U; panel < kN8Panels; ++panel) {
    state[panel] = zero_accumulator();
    score[panel] = zero_accumulator();
  }

#pragma unroll
  for (unsigned int key_panel = 0U; key_panel < kKeyPanels; ++key_panel) {
    const unsigned int first_key = key_panel * kBk;
#pragma unroll
    for (unsigned int pass = 0U; pass < 4U; ++pass) {
      const unsigned int copy = threadIdx.x + pass * kChunkThreads;
      const unsigned int row = copy / 8U;
      const unsigned int vector = copy & 7U;
      const unsigned int column = vector * 8U;
      cp_async_cg_16(shared_q + shared_index(row, column),
                     matrix_q + row * kKeyDimension + first_key + column);
      cp_async_cg_16(shared_k + shared_index(row, column),
                     matrix_k + row * kKeyDimension + first_key + column);
      cp_async_cg_16(
          shared_h + shared_index(row, column),
          matrix_h + (value_base + row) * kKeyDimension + first_key + column);
    }
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

#pragma unroll
    for (unsigned int k16 = 0U; k16 < kK16PerPanel; ++k16) {
      M16K16Fragment q_fragment{};
      load_m16k16(q_fragment, shared_q, query_panel, k16, lane);
#pragma unroll
      for (unsigned int panel = 0U; panel < kN8Panels; ++panel) {
        K16N8Fragment h_fragment{};
        load_k16n8_transposed(h_fragment, shared_h, panel, k16, lane);
        mma_m16n8k16(state[panel], q_fragment, h_fragment);
      }
#pragma unroll
      for (unsigned int panel = 0U; panel < kN8Panels; ++panel) {
        K16N8Fragment k_fragment{};
        load_k16n8_transposed(k_fragment, shared_k, panel, k16, lane);
        mma_m16n8k16(score[panel], q_fragment, k_fragment);
      }
    }
    __syncthreads();
  }

  // Q/K/H are dead. Keep the FP32 gate vector beside a 16 KiB QH layout
  // exchange so the second tensor-core phase can use the winning [2,2]
  // query/value warp ownership without any global intermediate.
  auto* const shared_qh = reinterpret_cast<float*>(shared_storage);
  auto* const shared_gamma = reinterpret_cast<float*>(
      shared_storage + kGammaOffset);
  if (threadIdx.x < kChunk) {
    shared_gamma[threadIdx.x] =
        cumulative_gate[value_matrix * kChunk + threadIdx.x];
  }
  __syncthreads();

  const unsigned int query0 = query_panel * 16U + lane / 4U;
  const unsigned int query1 = query0 + 8U;
  const float gate0 = expf(shared_gamma[query0]);
  const float gate1 = expf(shared_gamma[query1]);
#pragma unroll
  for (unsigned int panel = 0U; panel < kN8Panels; ++panel) {
    state[panel].x0 *= gate0;
    state[panel].x1 *= gate0;
    state[panel].x2 *= gate1;
    state[panel].x3 *= gate1;
    store_fp32_accumulator(shared_qh, state[panel], query_panel, panel, lane);
  }
  __syncthreads();

  const unsigned int query_group = warp >> 1U;
  const unsigned int value_group = warp & 1U;
#pragma unroll
  for (unsigned int m = 0U; m < 2U; ++m) {
#pragma unroll
    for (unsigned int n = 0U; n < 4U; ++n) {
      load_fp32_accumulator(state[m * 4U + n], shared_qh,
                            query_group * 2U + m,
                            value_group * 4U + n, lane);
    }
  }
  __syncthreads();

  auto* const shared_a = reinterpret_cast<std::uint16_t*>(
      shared_storage + kAOffset);
#pragma unroll
  for (unsigned int panel = 0U; panel < kN8Panels; ++panel) {
    store_gated_score_accumulator(shared_a, score[panel], query_panel,
                                  panel, lane, shared_gamma);
  }
  __syncthreads();

  // Global V is [source,value]. The matrix-B ldmatrix path wants its
  // canonical transpose [value,source], matching the exact Triton artifact.
  // Each thread performs four coalesced 16-byte loads and scatters only in
  // shared memory; there is no global transpose boundary.
  auto* const shared_v_transposed = reinterpret_cast<std::uint16_t*>(
      shared_storage + kVOffset);
#pragma unroll
  for (unsigned int pass = 0U; pass < 4U; ++pass) {
    const unsigned int copy = threadIdx.x + pass * kChunkThreads;
    const unsigned int source = copy / 8U;
    const unsigned int vector = copy & 7U;
    const uint4 values = *reinterpret_cast<const uint4*>(
        matrix_v + source * kValueDimension + value_base + vector * 8U);
    const unsigned int row = vector * 8U;
    shared_v_transposed[shared_index(row + 0U, source)] =
        static_cast<std::uint16_t>(values.x);
    shared_v_transposed[shared_index(row + 1U, source)] =
        static_cast<std::uint16_t>(values.x >> 16U);
    shared_v_transposed[shared_index(row + 2U, source)] =
        static_cast<std::uint16_t>(values.y);
    shared_v_transposed[shared_index(row + 3U, source)] =
        static_cast<std::uint16_t>(values.y >> 16U);
    shared_v_transposed[shared_index(row + 4U, source)] =
        static_cast<std::uint16_t>(values.z);
    shared_v_transposed[shared_index(row + 5U, source)] =
        static_cast<std::uint16_t>(values.z >> 16U);
    shared_v_transposed[shared_index(row + 6U, source)] =
        static_cast<std::uint16_t>(values.w);
    shared_v_transposed[shared_index(row + 7U, source)] =
        static_cast<std::uint16_t>(values.w >> 16U);
  }
  __syncthreads();

#pragma unroll
  for (unsigned int source_k16 = 0U; source_k16 < 4U; ++source_k16) {
    K16N8Fragment value_fragments[4U];
#pragma unroll
    for (unsigned int n = 0U; n < 4U; ++n) {
      load_k16n8_transposed(value_fragments[n], shared_v_transposed,
                            value_group * 4U + n, source_k16, lane);
    }
#pragma unroll
    for (unsigned int m = 0U; m < 2U; ++m) {
      M16K16Fragment a_fragment{};
      load_m16k16(a_fragment, shared_a, query_group * 2U + m,
                  source_k16, lane);
#pragma unroll
      for (unsigned int n = 0U; n < 4U; ++n) {
        mma_m16n8k16(state[m * 4U + n], a_fragment,
                     value_fragments[n]);
      }
    }
  }

#pragma unroll
  for (unsigned int m = 0U; m < 2U; ++m) {
#pragma unroll
    for (unsigned int n = 0U; n < 4U; ++n) {
      store_raw_output_accumulator(
          raw_output, state[m * 4U + n], token_base, value_head,
          value_base, query_group * 2U + m, value_group * 4U + n, lane);
    }
  }
}

__global__ __launch_bounds__(kNormThreads, 2)
void rms_norm_silu_rows8_kernel(
    const std::uint16_t* const raw_output,
    const std::uint16_t* const norm_weight,
    const std::uint16_t* const silu_gate,
    const unsigned int row_count,
    const float norm_epsilon,
    std::uint16_t* const output) {
  const unsigned int warp = threadIdx.x / 32U;
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int row = blockIdx.x * kNormRowsPerCta + warp;
  if (row >= row_count) {
    return;
  }
  const std::size_t base =
      static_cast<std::size_t>(row) * kValueDimension;
  const float value0 = decode_bf16(raw_output[base + lane]);
  const float value1 = decode_bf16(raw_output[base + lane + 32U]);
  const float value2 = decode_bf16(raw_output[base + lane + 64U]);
  const float value3 = decode_bf16(raw_output[base + lane + 96U]);
  float square_sum = fmaf(value0, value0, 0.0F);
  square_sum = fmaf(value1, value1, square_sum);
  square_sum = fmaf(value2, value2, square_sum);
  square_sum = fmaf(value3, value3, square_sum);
#pragma unroll
  for (unsigned int offset = 16U; offset != 0U; offset >>= 1U) {
    square_sum += __shfl_down_sync(0xffff'ffffU, square_sum, offset);
  }
  const float inverse_rms = __shfl_sync(
      0xffff'ffffU,
      rsqrtf(square_sum / static_cast<float>(kValueDimension) +
             norm_epsilon),
      0U);
  const float values[4U] = {value0, value1, value2, value3};
#pragma unroll
  for (unsigned int index = 0U; index < 4U; ++index) {
    const unsigned int value = lane + index * 32U;
    const float gate = decode_bf16(silu_gate[base + value]);
    const float normalized =
        values[index] * inverse_rms * decode_bf16(norm_weight[value]);
    output[base + value] = encode_bf16(
        normalized * gate / (1.0F + expf(-gate)));
  }
}

[[nodiscard]] bool invalid_arguments(
    const std::uint16_t* const compact_q,
    const std::uint16_t* const compact_k,
    const std::uint16_t* const boundary_state,
    const std::uint16_t* const v_new,
    const float* const cumulative_gate,
    const std::size_t token_count,
    const std::uint16_t* const norm_weight,
    const std::uint16_t* const silu_gate,
    const float norm_epsilon,
    const std::uint16_t* const raw_output,
    const std::uint16_t* const output) noexcept {
  return compact_q == nullptr || compact_k == nullptr ||
         boundary_state == nullptr || v_new == nullptr ||
         cumulative_gate == nullptr || token_count == 0U ||
         token_count > kMaximumTokens || token_count % kChunk != 0U ||
         norm_weight == nullptr || silu_gate == nullptr ||
         raw_output == nullptr || output == nullptr ||
         !std::isfinite(norm_epsilon) || norm_epsilon <= 0.0F;
}

template <typename Kernel>
[[nodiscard]] int query_kernel_resources(
    Kernel kernel, const int threads, int* const registers_per_thread,
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
  cudaError_t status = cudaFuncGetAttributes(&attributes, kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active, kernel, threads, 0U);
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

}  // namespace

int launch(const std::uint16_t* const compact_q,
           const std::uint16_t* const compact_k,
           const std::uint16_t* const boundary_state,
           const std::uint16_t* const v_new,
           const float* const cumulative_gate,
           const std::size_t token_count,
           const std::uint16_t* const norm_weight,
           const std::uint16_t* const silu_gate,
           const float norm_epsilon,
           std::uint16_t* const raw_output,
           std::uint16_t* const output,
           void* const cuda_stream) noexcept {
  if (invalid_arguments(compact_q, compact_k, boundary_state, v_new,
                        cumulative_gate, token_count, norm_weight, silu_gate,
                        norm_epsilon, raw_output, output)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const unsigned int chunk_count =
      static_cast<unsigned int>(token_count / kChunk);
  const dim3 grid(2U, chunk_count, kValueHeads);
  (void)cudaGetLastError();
  chunk_o_bv64_kernel<<<grid, kChunkThreads, 0U, stream>>>(
      compact_q, compact_k, boundary_state, v_new, cumulative_gate,
      raw_output);
  int status = static_cast<int>(cudaGetLastError());
  if (status != static_cast<int>(cudaSuccess)) {
    return status;
  }
  const unsigned int row_count =
      static_cast<unsigned int>(token_count * kRowsPerToken);
  const unsigned int norm_blocks =
      (row_count + kNormRowsPerCta - 1U) / kNormRowsPerCta;
  rms_norm_silu_rows8_kernel<<<norm_blocks, kNormThreads, 0U, stream>>>(
      raw_output, norm_weight, silu_gate, row_count, norm_epsilon, output);
  return static_cast<int>(cudaGetLastError());
}

int query_chunk_o_resources(int* const registers_per_thread,
                            std::size_t* const static_shared_bytes,
                            std::size_t* const local_bytes,
                            int* const maximum_threads_per_block,
                            int* const active_blocks_per_sm) noexcept {
  return query_kernel_resources(
      chunk_o_bv64_kernel, static_cast<int>(kChunkThreads),
      registers_per_thread, static_shared_bytes, local_bytes,
      maximum_threads_per_block, active_blocks_per_sm);
}

int query_norm_resources(int* const registers_per_thread,
                         std::size_t* const static_shared_bytes,
                         std::size_t* const local_bytes,
                         int* const maximum_threads_per_block,
                         int* const active_blocks_per_sm) noexcept {
  return query_kernel_resources(
      rms_norm_silu_rows8_kernel, static_cast<int>(kNormThreads),
      registers_per_thread, static_shared_bytes, local_bytes,
      maximum_threads_per_block, active_blocks_per_sm);
}

}  // namespace q3x::runtime::gdn_prefill_chunk_o_bv64_detail
