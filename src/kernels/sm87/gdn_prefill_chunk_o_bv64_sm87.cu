#include "gdn_prefill_chunk_o_bv64_sm87.h"

#include "q3x/kernels/sm87_a4w4_attention_k256_m128n256.h"
#include "q3x/kernels/sm87_a4w4_factorized_lane_quantize.h"
#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::runtime::gdn_prefill_chunk_o_bv64_detail {
namespace {

namespace wmma = nvcuda::wmma;
using Bf16 = __nv_bfloat16;
using WmmaAccumulator =
    wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

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
constexpr unsigned int kSharedLeadingDimension = 64U;
constexpr unsigned int kN8Panels = kBv / 8U;
constexpr unsigned int kK16PerPanel = kBk / 16U;
constexpr unsigned int kKeyPanels = kKeyDimension / kBk;
constexpr unsigned int kSharedTileElements =
    kChunk * kSharedLeadingDimension;
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
constexpr unsigned int kMaximumNormTokens = 4'096U;
constexpr unsigned int kK256A4InputSize =
    kValueHeads * kValueDimension;
constexpr unsigned int kK256A4Groups =
    kK256A4InputSize /
    q3x::kernels::kSm87A4W4AttentionK256ScaleK;
constexpr unsigned int kK256A4PhysicalGroups =
    kK256A4InputSize /
    q3x::kernels::kSm87A4W4AttentionK256PhysicalK64;
constexpr unsigned int kHeadOctetsPerToken =
    kValueHeads / kNormRowsPerCta;
constexpr unsigned int kMaximumNormK256A4Registers = 128U;
constexpr unsigned int kMinimumNormK256A4CtasPerSm = 2U;
constexpr std::size_t kNormK256A4SharedBytes =
    kNormRowsPerCta * sizeof(float);
constexpr unsigned int kFactorizedLaneR1LaneCount = 1U;
constexpr unsigned int kFactorizedLaneR1Pairs = kK256A4InputSize / 2U;
constexpr unsigned int kFactorizedLaneR1HeadWaves =
    kValueHeads / (kNormThreads / 32U);
constexpr unsigned int kMaximumNormFactorizedLaneR1A4Registers = 128U;
constexpr unsigned int kMinimumNormFactorizedLaneR1A4CtasPerSm = 2U;

struct alignas(16) FactorizedLaneR1SharedStorage final {
  // The exact normalized BF16 seam is CTA-private: it is consumed once by
  // the whole-K6144 quantizer and never published to a runner-owned tensor.
  std::uint16_t seam[kK256A4InputSize];
  float inverse_rms[kValueHeads];
  float warp_maxima[kNormThreads / 32U];
  float clipped_maximum;
  float stored_scale;
  std::uint16_t scale_bits;
  std::uint16_t reserved;
};

constexpr std::size_t kNormFactorizedLaneR1A4SharedBytes =
    sizeof(FactorizedLaneR1SharedStorage);

static_assert(kQkHeads * kHeadGroup == kValueHeads);
static_assert(kChunkWarps == 4U);
static_assert(kNormRowsPerCta == 8U);
static_assert(kSharedTileBytes == 8'192U);
static_assert(kSharedBytes == 24'576U);
static_assert(kK256A4InputSize == 6'144U);
static_assert(kK256A4Groups == 24U);
static_assert(kK256A4PhysicalGroups == 96U);
static_assert(kHeadOctetsPerToken == 6U);
static_assert(kNormK256A4SharedBytes == 32U);
static_assert(kFactorizedLaneR1Pairs == 3'072U);
static_assert(kFactorizedLaneR1HeadWaves == 6U);
static_assert(kNormFactorizedLaneR1A4SharedBytes == 12'528U);

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
  return row * kSharedLeadingDimension +
         (column ^ ((row & 7U) * 8U));
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
  // tile is a true [K,N] row-major backing for the logical column-major B
  // operand. Lanes 0..15 provide the sixteen K rows of one K16xN8 tile;
  // ldmatrix.trans then emits the official mma.m16n8k16 B lane fragment.
  const unsigned int row =
      k16 * 16U + (lane & 7U) + ((lane >> 3U) & 1U) * 8U;
  const unsigned int column = n_panel * 8U;
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

// Preserve the Attention K256 quantizer's bit-level BF16 RNE semantics.
// Unlike the GDN boundary encoder above, this intentionally does not
// canonicalize NaNs.
[[nodiscard]] __device__ __forceinline__ std::uint16_t
encode_quantizer_bf16(const float value) {
  unsigned int bits = __float_as_uint(value);
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

[[nodiscard]] __device__ __forceinline__ float warp_maximum(
    float value) {
#pragma unroll
  for (unsigned int offset = 16U; offset != 0U; offset >>= 1U) {
    value = fmaxf(
        value, __shfl_down_sync(0xffff'ffffU, value, offset));
  }
  return value;
}

[[nodiscard]] __device__ __forceinline__ std::uint16_t
normalized_silu_gate_bf16(
    const std::uint16_t raw_bits, const float inverse_rms,
    const std::uint16_t norm_weight_bits,
    const std::uint16_t gate_bits) {
  const float raw = decode_bf16(raw_bits);
  const float gate = decode_bf16(gate_bits);
  const float normalized =
      raw * inverse_rms * decode_bf16(norm_weight_bits);
  return encode_bf16(normalized * gate / (1.0F + expf(-gate)));
}

[[nodiscard]] __device__ __forceinline__ std::uint32_t pack_bf16_pair(
    const std::uint16_t low, const std::uint16_t high) {
  return static_cast<std::uint32_t>(low) |
         (static_cast<std::uint32_t>(high) << 16U);
}

__device__ __forceinline__ void store_bf16_vector_nk_to_kn(
    std::uint16_t* const destination, const unsigned int first_key,
    const unsigned int n, const uint4 values) {
  destination[shared_index(first_key + 0U, n)] =
      static_cast<std::uint16_t>(values.x);
  destination[shared_index(first_key + 1U, n)] =
      static_cast<std::uint16_t>(values.x >> 16U);
  destination[shared_index(first_key + 2U, n)] =
      static_cast<std::uint16_t>(values.y);
  destination[shared_index(first_key + 3U, n)] =
      static_cast<std::uint16_t>(values.y >> 16U);
  destination[shared_index(first_key + 4U, n)] =
      static_cast<std::uint16_t>(values.z);
  destination[shared_index(first_key + 5U, n)] =
      static_cast<std::uint16_t>(values.z >> 16U);
  destination[shared_index(first_key + 6U, n)] =
      static_cast<std::uint16_t>(values.w);
  destination[shared_index(first_key + 7U, n)] =
      static_cast<std::uint16_t>(values.w >> 16U);
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

__global__ __launch_bounds__(32)
void fragment_lane_sentinel_kernel(
    const std::uint16_t* const matrix_a,
    const std::uint16_t* const matrix_b,
    std::uint32_t* const loaded_a_output,
    std::uint32_t* const direct_a_output,
    std::uint32_t* const loaded_b_output,
    std::uint32_t* const direct_b_output,
    float* const loaded_accumulator_output,
    float* const direct_accumulator_output) {
  __shared__ std::uint16_t shared_a[16U * kSharedLeadingDimension];
  __shared__ std::uint16_t shared_b[16U * kSharedLeadingDimension];
  const unsigned int lane = threadIdx.x;
  for (unsigned int index = lane; index < 16U * 16U; index += 32U) {
    const unsigned int row = index / 16U;
    const unsigned int column = index % 16U;
    shared_a[shared_index(row, column)] = matrix_a[index];
    shared_b[shared_index(column, row)] = matrix_b[index];
  }
  __syncwarp();

  M16K16Fragment loaded_a{};
  K16N8Fragment loaded_b{};
  load_m16k16(loaded_a, shared_a, 0U, 0U, lane);
  load_k16n8_transposed(loaded_b, shared_b, 0U, 0U, lane);

  const unsigned int group = lane / 4U;
  const unsigned int thread_in_group = lane & 3U;
  const unsigned int low_key = thread_in_group * 2U;
  const M16K16Fragment direct_a{
      pack_bf16_pair(shared_a[shared_index(group, low_key)],
                     shared_a[shared_index(group, low_key + 1U)]),
      pack_bf16_pair(shared_a[shared_index(group + 8U, low_key)],
                     shared_a[shared_index(group + 8U, low_key + 1U)]),
      pack_bf16_pair(shared_a[shared_index(group, low_key + 8U)],
                     shared_a[shared_index(group, low_key + 9U)]),
      pack_bf16_pair(shared_a[shared_index(group + 8U, low_key + 8U)],
                     shared_a[shared_index(group + 8U, low_key + 9U)])};
  const K16N8Fragment direct_b{
      pack_bf16_pair(shared_b[shared_index(low_key, group)],
                     shared_b[shared_index(low_key + 1U, group)]),
      pack_bf16_pair(shared_b[shared_index(low_key + 8U, group)],
                     shared_b[shared_index(low_key + 9U, group)])};

  loaded_a_output[lane * 4U + 0U] = loaded_a.x0;
  loaded_a_output[lane * 4U + 1U] = loaded_a.x1;
  loaded_a_output[lane * 4U + 2U] = loaded_a.x2;
  loaded_a_output[lane * 4U + 3U] = loaded_a.x3;
  direct_a_output[lane * 4U + 0U] = direct_a.x0;
  direct_a_output[lane * 4U + 1U] = direct_a.x1;
  direct_a_output[lane * 4U + 2U] = direct_a.x2;
  direct_a_output[lane * 4U + 3U] = direct_a.x3;
  loaded_b_output[lane * 2U + 0U] = loaded_b.x0;
  loaded_b_output[lane * 2U + 1U] = loaded_b.x1;
  direct_b_output[lane * 2U + 0U] = direct_b.x0;
  direct_b_output[lane * 2U + 1U] = direct_b.x1;

  M16N8Accumulator loaded_accumulator = zero_accumulator();
  M16N8Accumulator direct_accumulator = zero_accumulator();
  mma_m16n8k16(loaded_accumulator, loaded_a, loaded_b);
  mma_m16n8k16(direct_accumulator, direct_a, direct_b);
  loaded_accumulator_output[lane * 4U + 0U] = loaded_accumulator.x0;
  loaded_accumulator_output[lane * 4U + 1U] = loaded_accumulator.x1;
  loaded_accumulator_output[lane * 4U + 2U] = loaded_accumulator.x2;
  loaded_accumulator_output[lane * 4U + 3U] = loaded_accumulator.x3;
  direct_accumulator_output[lane * 4U + 0U] = direct_accumulator.x0;
  direct_accumulator_output[lane * 4U + 1U] = direct_accumulator.x1;
  direct_accumulator_output[lane * 4U + 2U] = direct_accumulator.x2;
  direct_accumulator_output[lane * 4U + 3U] = direct_accumulator.x3;
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
      const uint4 k_values = *reinterpret_cast<const uint4*>(
          matrix_k + row * kKeyDimension + first_key + column);
      const uint4 h_values = *reinterpret_cast<const uint4*>(
          matrix_h + (value_base + row) * kKeyDimension + first_key + column);
      store_bf16_vector_nk_to_kn(shared_k, column, row, k_values);
      store_bf16_vector_nk_to_kn(shared_h, column, row, h_values);
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

  // Global V already is the true [K=source,N=value] backing required by the
  // transposed matrix-B ldmatrix path. Stage it coalesced without the former
  // register-to-shared transpose.
  auto* const shared_v = reinterpret_cast<std::uint16_t*>(
      shared_storage + kVOffset);
#pragma unroll
  for (unsigned int pass = 0U; pass < 4U; ++pass) {
    const unsigned int copy = threadIdx.x + pass * kChunkThreads;
    const unsigned int source = copy / 8U;
    const unsigned int vector = copy & 7U;
    const unsigned int first_value = vector * 8U;
    cp_async_cg_16(
        shared_v + shared_index(source, first_value),
        matrix_v + source * kValueDimension + value_base + first_value);
  }
  cp_async_commit();
  cp_async_wait_all();
  __syncthreads();

#pragma unroll
  for (unsigned int source_k16 = 0U; source_k16 < 4U; ++source_k16) {
    K16N8Fragment value_fragments[4U];
#pragma unroll
    for (unsigned int n = 0U; n < 4U; ++n) {
      load_k16n8_transposed(value_fragments[n], shared_v,
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

// Correctness-only oracle for the inline-PTX fragment mapping above.  It
// retains the same BV64 CTA/output ownership and the same BF16 QK boundary,
// but delegates fragment packing to CUDA WMMA.  It is never selected by the
// engine and is intentionally excluded from performance admission.
__global__ __launch_bounds__(kChunkThreads)
void chunk_o_bv64_wmma_oracle_kernel(
    const std::uint16_t* const compact_q,
    const std::uint16_t* const compact_k,
    const std::uint16_t* const boundary_state,
    const std::uint16_t* const v_new,
    const float* const cumulative_gate,
    std::uint16_t* const raw_output) {
  __shared__ __align__(16) unsigned char shared_storage[kSharedBytes];
  auto* const shared_q = reinterpret_cast<Bf16*>(
      shared_storage + kQOffset);
  auto* const shared_k = reinterpret_cast<Bf16*>(
      shared_storage + kKOffset);
  auto* const shared_h = reinterpret_cast<Bf16*>(
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
  const auto* const matrix_q = reinterpret_cast<const Bf16*>(
      compact_q + compact_matrix * kChunk * kKeyDimension);
  const auto* const matrix_k = reinterpret_cast<const Bf16*>(
      compact_k + compact_matrix * kChunk * kKeyDimension);
  const auto* const matrix_h = reinterpret_cast<const Bf16*>(
      boundary_state + value_matrix * kValueDimension * kKeyDimension);
  const auto* const matrix_v = reinterpret_cast<const Bf16*>(
      v_new + value_matrix * kChunk * kValueDimension);

  const unsigned int warp = threadIdx.x / 32U;
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int query_panel = warp;

  WmmaAccumulator state[4U];
  WmmaAccumulator score[4U];
#pragma unroll
  for (unsigned int panel = 0U; panel < 4U; ++panel) {
    wmma::fill_fragment(state[panel], 0.0F);
    wmma::fill_fragment(score[panel], 0.0F);
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
      cp_async_cg_16(shared_q + row * kBk + column,
                     matrix_q + row * kKeyDimension + first_key + column);
      cp_async_cg_16(shared_k + row * kBk + column,
                     matrix_k + row * kKeyDimension + first_key + column);
      cp_async_cg_16(
          shared_h + row * kBk + column,
          matrix_h + (value_base + row) * kKeyDimension + first_key + column);
    }
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

#pragma unroll
    for (unsigned int k16 = 0U; k16 < kK16PerPanel; ++k16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, Bf16,
                     wmma::row_major>
          q_fragment;
      wmma::load_matrix_sync(
          q_fragment,
          shared_q + query_panel * 16U * kBk + k16 * 16U,
          static_cast<int>(kBk));
#pragma unroll
      for (unsigned int panel = 0U; panel < 4U; ++panel) {
        wmma::fragment<wmma::matrix_b, 16, 16, 16, Bf16,
                       wmma::col_major>
            h_fragment;
        wmma::load_matrix_sync(
            h_fragment, shared_h + panel * 16U * kBk + k16 * 16U,
            static_cast<int>(kBk));
        wmma::mma_sync(state[panel], q_fragment, h_fragment,
                       state[panel]);

        wmma::fragment<wmma::matrix_b, 16, 16, 16, Bf16,
                       wmma::col_major>
            k_fragment;
        wmma::load_matrix_sync(
            k_fragment, shared_k + panel * 16U * kBk + k16 * 16U,
            static_cast<int>(kBk));
        wmma::mma_sync(score[panel], q_fragment, k_fragment,
                       score[panel]);
      }
    }
    __syncthreads();
  }

  auto* const shared_qh = reinterpret_cast<float*>(shared_storage);
  auto* const shared_gamma = reinterpret_cast<float*>(
      shared_storage + kGammaOffset);
  if (threadIdx.x < kChunk) {
    shared_gamma[threadIdx.x] =
        cumulative_gate[value_matrix * kChunk + threadIdx.x];
  }
  __syncthreads();

#pragma unroll
  for (unsigned int panel = 0U; panel < 4U; ++panel) {
    float* const tile =
        shared_qh + query_panel * 16U * kBv + panel * 16U;
    wmma::store_matrix_sync(tile, state[panel], static_cast<int>(kBv),
                            wmma::mem_row_major);
    __syncwarp();
    for (unsigned int index = lane; index < 16U * 16U; index += 32U) {
      const unsigned int query = query_panel * 16U + index / 16U;
      tile[(index / 16U) * kBv + index % 16U] *=
          expf(shared_gamma[query]);
    }
  }
  __syncthreads();

  const unsigned int query_group = warp >> 1U;
  const unsigned int value_group = warp & 1U;
  WmmaAccumulator output_accumulators[4U];
#pragma unroll
  for (unsigned int m = 0U; m < 2U; ++m) {
#pragma unroll
    for (unsigned int n = 0U; n < 2U; ++n) {
      wmma::load_matrix_sync(
          output_accumulators[m * 2U + n],
          shared_qh + (query_group * 32U + m * 16U) * kBv +
              value_group * 32U + n * 16U,
          static_cast<int>(kBv), wmma::mem_row_major);
    }
  }
  __syncthreads();

  auto* const shared_a = reinterpret_cast<Bf16*>(
      shared_storage + kAOffset);
  auto* const score_scratch = reinterpret_cast<float*>(
      shared_storage + kVOffset) + warp * 16U * 16U;
#pragma unroll
  for (unsigned int panel = 0U; panel < 4U; ++panel) {
    wmma::store_matrix_sync(score_scratch, score[panel], 16,
                            wmma::mem_row_major);
    __syncwarp();
    for (unsigned int index = lane; index < 16U * 16U; index += 32U) {
      const unsigned int query = query_panel * 16U + index / 16U;
      const unsigned int source = panel * 16U + index % 16U;
      shared_a[query * kChunk + source] = __float2bfloat16_rn(
          source <= query
              ? score_scratch[index] *
                    expf(shared_gamma[query] - shared_gamma[source])
              : 0.0F);
    }
  }
  __syncthreads();

#pragma unroll
  for (unsigned int source_panel = 0U; source_panel < 4U;
       ++source_panel) {
#pragma unroll
    for (unsigned int m = 0U; m < 2U; ++m) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, Bf16,
                     wmma::row_major>
          a_fragment;
      wmma::load_matrix_sync(
          a_fragment,
          shared_a + (query_group * 32U + m * 16U) * kChunk +
              source_panel * 16U,
          static_cast<int>(kChunk));
#pragma unroll
      for (unsigned int n = 0U; n < 2U; ++n) {
        wmma::fragment<wmma::matrix_b, 16, 16, 16, Bf16,
                       wmma::row_major>
            v_fragment;
        wmma::load_matrix_sync(
            v_fragment,
            matrix_v + source_panel * 16U * kValueDimension + value_base +
                value_group * 32U + n * 16U,
            static_cast<int>(kValueDimension));
        wmma::mma_sync(output_accumulators[m * 2U + n], a_fragment,
                       v_fragment, output_accumulators[m * 2U + n]);
      }
    }
  }

  __syncthreads();
  auto* const output_scratch = reinterpret_cast<float*>(shared_storage) +
                               warp * 16U * 16U;
#pragma unroll
  for (unsigned int m = 0U; m < 2U; ++m) {
#pragma unroll
    for (unsigned int n = 0U; n < 2U; ++n) {
      wmma::store_matrix_sync(output_scratch,
                              output_accumulators[m * 2U + n], 16,
                              wmma::mem_row_major);
      __syncwarp();
      for (unsigned int index = lane; index < 16U * 16U; index += 32U) {
        const unsigned int query =
            query_group * 32U + m * 16U + index / 16U;
        const unsigned int value =
            value_base + value_group * 32U + n * 16U + index % 16U;
        raw_output[
            static_cast<std::size_t>(token_base + query) * kValuesPerToken +
            value_head * kValueDimension + value] =
            __bfloat16_as_ushort(__float2bfloat16_rn(output_scratch[index]));
      }
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

__global__ __launch_bounds__(kNormThreads, kMinimumNormK256A4CtasPerSm)
void rms_norm_silu_rows8_k256_a4_kernel(
    const std::uint16_t* const raw_output_tile,
    const std::uint16_t* const norm_weight,
    const std::uint16_t* const silu_gate_tile,
    const unsigned int tile_logical_token_count,
    const float norm_epsilon,
    const unsigned int destination_first_token,
    const unsigned int publish_token_count,
    const float clip_ratio,
    std::uint8_t* const packed_a_base,
    std::uint16_t* const scales_bf16_base) {
  __shared__ float head_maximum[kNormRowsPerCta];
  const unsigned int warp = threadIdx.x / 32U;
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int local_token = blockIdx.x / kHeadOctetsPerToken;
  const unsigned int head_octet = blockIdx.x % kHeadOctetsPerToken;
  const unsigned int head = head_octet * kNormRowsPerCta + warp;
  const std::size_t destination_token =
      static_cast<std::size_t>(destination_first_token) + local_token;

  if (local_token >= publish_token_count) {
    return;
  }
  if (local_token >= tile_logical_token_count) {
    // Only the final slice can enter this uniform CTA branch.  Each warp
    // owns the two physical K64 code planes of one D128 head; the even warp
    // in every adjacent pair owns their K256 scale.
#pragma unroll
    for (unsigned int index = 0U; index < 4U; ++index) {
      if ((lane & 1U) == 0U) {
        const std::size_t physical_group =
            static_cast<std::size_t>(head) * 2U + index / 2U;
        const std::size_t byte_in_k64 =
            (index & 1U) * 16U + lane / 2U;
        packed_a_base[
            q3x::kernels::sm87_a4w4_attention_k256_packed_offset(
                destination_token, physical_group, byte_in_k64,
                kK256A4PhysicalGroups)] = 0U;
      }
    }
    if ((warp & 1U) == 0U && lane == 0U) {
      scales_bf16_base[
          q3x::kernels::sm87_a4w4_attention_k256_scale_offset(
              destination_token, head / 2U, kK256A4Groups)] = 0x3f80U;
    }
    return;
  }

  const std::size_t input_base =
      (static_cast<std::size_t>(local_token) * kValueHeads + head) *
      kValueDimension;
  const float raw_values[4U] = {
      decode_bf16(raw_output_tile[input_base + lane]),
      decode_bf16(raw_output_tile[input_base + lane + 32U]),
      decode_bf16(raw_output_tile[input_base + lane + 64U]),
      decode_bf16(raw_output_tile[input_base + lane + 96U])};
  float square_sum = fmaf(raw_values[0U], raw_values[0U], 0.0F);
  square_sum = fmaf(raw_values[1U], raw_values[1U], square_sum);
  square_sum = fmaf(raw_values[2U], raw_values[2U], square_sum);
  square_sum = fmaf(raw_values[3U], raw_values[3U], square_sum);
#pragma unroll
  for (unsigned int offset = 16U; offset != 0U; offset >>= 1U) {
    square_sum += __shfl_down_sync(0xffff'ffffU, square_sum, offset);
  }
  const float inverse_rms = __shfl_sync(
      0xffff'ffffU,
      rsqrtf(square_sum / static_cast<float>(kValueDimension) +
             norm_epsilon),
      0U);

  float values[4U];
  float maximum = 0.0F;
#pragma unroll
  for (unsigned int index = 0U; index < 4U; ++index) {
    const unsigned int value = lane + index * 32U;
    const float gate = decode_bf16(silu_gate_tile[input_base + value]);
    const float normalized =
        raw_values[index] * inverse_rms * decode_bf16(norm_weight[value]);
    // This round trip is the observable incumbent seam between the GDN
    // epilogue and the standalone K256 quantizer.
    values[index] = decode_bf16(encode_bf16(
        normalized * gate / (1.0F + expf(-gate))));
    maximum = fmaxf(maximum, fabsf(values[index]));
  }
#pragma unroll
  for (unsigned int offset = 16U; offset != 0U; offset >>= 1U) {
    maximum =
        fmaxf(maximum,
              __shfl_down_sync(0xffff'ffffU, maximum, offset));
  }
  if (lane == 0U) {
    head_maximum[warp] = maximum;
  }
  __syncthreads();

  maximum = fmaxf(head_maximum[warp & ~1U],
                  head_maximum[warp | 1U]);
  const float clipped_maximum = maximum * clip_ratio;
  std::uint16_t scale_bits = encode_quantizer_bf16(
      maximum == 0.0F ? 1.0F : clipped_maximum / 7.0F);
  float stored_scale = decode_bf16(scale_bits);
  if (maximum != 0.0F && stored_scale == 0.0F) {
    scale_bits = 1U;
    stored_scale = decode_bf16(scale_bits);
  }

  if ((warp & 1U) == 0U && lane == 0U) {
    scales_bf16_base[
        q3x::kernels::sm87_a4w4_attention_k256_scale_offset(
            destination_token, head / 2U, kK256A4Groups)] = scale_bits;
  }
#pragma unroll
  for (unsigned int index = 0U; index < 4U; ++index) {
    const float even_value = values[index];
    const float odd_value =
        __shfl_xor_sync(0xffff'ffffU, even_value, 1U);
    if ((lane & 1U) == 0U) {
      const float even =
          fminf(fmaxf(even_value, -clipped_maximum), clipped_maximum);
      const float odd =
          fminf(fmaxf(odd_value, -clipped_maximum), clipped_maximum);
      const int even_rounded = stored_scale == 0.0F
                                   ? 0
                                   : __float2int_rn(even / stored_scale);
      const int odd_rounded = stored_scale == 0.0F
                                  ? 0
                                  : __float2int_rn(odd / stored_scale);
      const int even_code = even_rounded < -7
                                ? -7
                                : even_rounded > 7 ? 7 : even_rounded;
      const int odd_code = odd_rounded < -7
                               ? -7
                               : odd_rounded > 7 ? 7 : odd_rounded;
      const std::size_t physical_group =
          static_cast<std::size_t>(head) * 2U + index / 2U;
      const std::size_t byte_in_k64 =
          (index & 1U) * 16U + lane / 2U;
      packed_a_base[
          q3x::kernels::sm87_a4w4_attention_k256_packed_offset(
              destination_token, physical_group, byte_in_k64,
              kK256A4PhysicalGroups)] =
          q3x::kernels::sm87_a4w4_pack_signed_pair(
              even_code, odd_code);
    }
  }
}

__global__ __launch_bounds__(
    kNormThreads, kMinimumNormFactorizedLaneR1A4CtasPerSm)
void rms_norm_silu_factorized_lane_r1_a4_kernel(
    const std::uint16_t* const raw_output_tile,
    const std::uint16_t* const norm_weight,
    const std::uint16_t* const silu_gate_tile,
    const unsigned int tile_logical_token_count,
    const float norm_epsilon,
    const float* const inverse_alpha_fp32,
    const unsigned int destination_first_token,
    const unsigned int publish_token_count,
    const float clip_ratio,
    std::uint8_t* const packed_a_base,
    std::uint16_t* const scales_bf16_base) {
  __shared__ FactorizedLaneR1SharedStorage shared;
  const unsigned int thread = threadIdx.x;
  const unsigned int warp = thread >> 5U;
  const unsigned int lane = thread & 31U;
  const unsigned int local_token = blockIdx.x;
  const std::size_t destination_token =
      static_cast<std::size_t>(destination_first_token) + local_token;

  if (local_token >= publish_token_count) {
    return;
  }
  if (local_token >= tile_logical_token_count) {
    // Only the final state slice owns padded rows.  It never reads the
    // tile-local raw/gate tensors or the authenticated inverse-alpha plane.
    for (unsigned int pair = thread; pair < kFactorizedLaneR1Pairs;
         pair += kNormThreads) {
      const unsigned int global_even = pair << 1U;
      packed_a_base[q3x::kernels::sm87_a4w4_consumer_packed_offset(
          destination_token, global_even >> 6U,
          (global_even & 63U) >> 1U, kK256A4PhysicalGroups)] = 0U;
    }
    if (thread == 0U) {
      scales_bf16_base[
          q3x::kernels::sm87_a4w4_factorized_lane_scale_offset(
              destination_token, 0U, kFactorizedLaneR1LaneCount)] = 0x3f80U;
    }
    return;
  }

  const std::size_t token_base =
      static_cast<std::size_t>(local_token) * kK256A4InputSize;

  // Preserve the incumbent rows-8 numerical ownership exactly: one warp
  // reduces one D128 head with four values per lane.  Eight warps cover all
  // 48 heads in six waves before any normalized value is produced.
#pragma unroll
  for (unsigned int wave = 0U; wave < kFactorizedLaneR1HeadWaves; ++wave) {
    const unsigned int head = wave * (kNormThreads / 32U) + warp;
    const std::size_t head_base =
        token_base + static_cast<std::size_t>(head) * kValueDimension;
    const float value0 = decode_bf16(raw_output_tile[head_base + lane]);
    const float value1 =
        decode_bf16(raw_output_tile[head_base + lane + 32U]);
    const float value2 =
        decode_bf16(raw_output_tile[head_base + lane + 64U]);
    const float value3 =
        decode_bf16(raw_output_tile[head_base + lane + 96U]);
    float square_sum = fmaf(value0, value0, 0.0F);
    square_sum = fmaf(value1, value1, square_sum);
    square_sum = fmaf(value2, value2, square_sum);
    square_sum = fmaf(value3, value3, square_sum);
#pragma unroll
    for (unsigned int offset = 16U; offset != 0U; offset >>= 1U) {
      square_sum +=
          __shfl_down_sync(0xffff'ffffU, square_sum, offset);
    }
    if (lane == 0U) {
      shared.inverse_rms[head] = rsqrtf(
          square_sum / static_cast<float>(kValueDimension) + norm_epsilon);
    }
  }
  __syncthreads();

  // Materialize only the established BF16 seam in shared memory.  Pair-wise
  // uint32 stores avoid the two-way bank conflict of 16-bit warp stores.
  // The same values feed both the global R1 maximum and the final codes.
  auto* const seam_pairs =
      reinterpret_cast<std::uint32_t*>(shared.seam);
  float maximum = 0.0F;
  for (unsigned int pair = thread; pair < kFactorizedLaneR1Pairs;
       pair += kNormThreads) {
    const unsigned int global_even = pair << 1U;
    const unsigned int head = global_even / kValueDimension;
    const unsigned int dimension = global_even % kValueDimension;
    const std::size_t input_even = token_base + global_even;
    const std::uint16_t even_bits = normalized_silu_gate_bf16(
        raw_output_tile[input_even], shared.inverse_rms[head],
        norm_weight[dimension], silu_gate_tile[input_even]);
    const std::uint16_t odd_bits = normalized_silu_gate_bf16(
        raw_output_tile[input_even + 1U], shared.inverse_rms[head],
        norm_weight[dimension + 1U], silu_gate_tile[input_even + 1U]);
    seam_pairs[pair] = pack_bf16_pair(even_bits, odd_bits);
    maximum = fmaxf(
        maximum,
        fabsf(decode_bf16(even_bits) * inverse_alpha_fp32[global_even]));
    maximum = fmaxf(
        maximum,
        fabsf(decode_bf16(odd_bits) *
              inverse_alpha_fp32[global_even + 1U]));
  }

  maximum = warp_maximum(maximum);
  if (lane == 0U) {
    shared.warp_maxima[warp] = maximum;
  }
  __syncthreads();
  if (warp == 0U) {
    float block_maximum =
        lane < kNormThreads / 32U ? shared.warp_maxima[lane] : 0.0F;
    block_maximum = warp_maximum(block_maximum);
    if (lane == 0U) {
      const float clipped_maximum = block_maximum * clip_ratio;
      std::uint16_t scale_bits = encode_quantizer_bf16(
          block_maximum == 0.0F ? 1.0F : clipped_maximum / 7.0F);
      float stored_scale = decode_bf16(scale_bits);
      if (block_maximum != 0.0F && stored_scale == 0.0F) {
        scale_bits = 1U;
        stored_scale = decode_bf16(scale_bits);
      }
      shared.clipped_maximum = clipped_maximum;
      shared.stored_scale = stored_scale;
      shared.scale_bits = scale_bits;
    }
  }
  __syncthreads();

  for (unsigned int pair = thread; pair < kFactorizedLaneR1Pairs;
       pair += kNormThreads) {
    const unsigned int global_even = pair << 1U;
    const std::uint32_t seam = seam_pairs[pair];
    float even = decode_bf16(static_cast<std::uint16_t>(seam)) *
                 inverse_alpha_fp32[global_even];
    float odd = decode_bf16(static_cast<std::uint16_t>(seam >> 16U)) *
                inverse_alpha_fp32[global_even + 1U];
    even = fminf(fmaxf(even, -shared.clipped_maximum),
                 shared.clipped_maximum);
    odd = fminf(fmaxf(odd, -shared.clipped_maximum),
                shared.clipped_maximum);
    const int even_rounded = __float2int_rn(even / shared.stored_scale);
    const int odd_rounded = __float2int_rn(odd / shared.stored_scale);
    const int even_code = even_rounded < -7
                              ? -7
                              : (even_rounded > 7 ? 7 : even_rounded);
    const int odd_code = odd_rounded < -7
                             ? -7
                             : (odd_rounded > 7 ? 7 : odd_rounded);
    packed_a_base[q3x::kernels::sm87_a4w4_consumer_packed_offset(
        destination_token, global_even >> 6U,
        (global_even & 63U) >> 1U, kK256A4PhysicalGroups)] =
        q3x::kernels::sm87_a4w4_pack_signed_pair(even_code, odd_code);
  }
  if (thread == 0U) {
    scales_bf16_base[
        q3x::kernels::sm87_a4w4_factorized_lane_scale_offset(
            destination_token, 0U, kFactorizedLaneR1LaneCount)] =
        shared.scale_bits;
  }
}

[[nodiscard]] constexpr bool aligned(
    const void* const pointer, const std::size_t alignment) noexcept {
  return pointer != nullptr &&
         reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0U;
}

[[nodiscard]] bool byte_ranges_overlap(
    const void* const first, const std::size_t first_bytes,
    const void* const second, const std::size_t second_bytes) noexcept {
  const std::uintptr_t first_begin =
      reinterpret_cast<std::uintptr_t>(first);
  const std::uintptr_t second_begin =
      reinterpret_cast<std::uintptr_t>(second);
  constexpr std::uintptr_t maximum =
      std::numeric_limits<std::uintptr_t>::max();
  if (first_bytes > maximum - first_begin ||
      second_bytes > maximum - second_begin) {
    return true;
  }
  const std::uintptr_t first_end = first_begin + first_bytes;
  const std::uintptr_t second_end = second_begin + second_bytes;
  return first_begin < second_end && second_begin < first_end;
}

[[nodiscard]] bool invalid_raw_arguments(
    const std::uint16_t* const compact_q,
    const std::uint16_t* const compact_k,
    const std::uint16_t* const boundary_state,
    const std::uint16_t* const v_new,
    const float* const cumulative_gate,
    const std::size_t token_count,
    const std::uint16_t* const raw_output) noexcept {
  return compact_q == nullptr || compact_k == nullptr ||
         boundary_state == nullptr || v_new == nullptr ||
         cumulative_gate == nullptr || token_count == 0U ||
         token_count > kMaximumTokens || raw_output == nullptr;
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
  return invalid_raw_arguments(compact_q, compact_k, boundary_state, v_new,
                               cumulative_gate, token_count, raw_output) ||
         norm_weight == nullptr || silu_gate == nullptr ||
         output == nullptr ||
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

int launch_raw(const std::uint16_t* const compact_q,
               const std::uint16_t* const compact_k,
               const std::uint16_t* const boundary_state,
               const std::uint16_t* const v_new,
               const float* const cumulative_gate,
               const std::size_t token_count,
               std::uint16_t* const raw_output,
               void* const cuda_stream) noexcept {
  if (invalid_raw_arguments(compact_q, compact_k, boundary_state, v_new,
                            cumulative_gate, token_count, raw_output)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const unsigned int chunk_count =
      static_cast<unsigned int>((token_count + kChunk - 1U) / kChunk);
  const dim3 grid(2U, chunk_count, kValueHeads);
  (void)cudaGetLastError();
  chunk_o_bv64_kernel<<<grid, kChunkThreads, 0U, stream>>>(
      compact_q, compact_k, boundary_state, v_new, cumulative_gate,
      raw_output);
  return static_cast<int>(cudaGetLastError());
}

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
  const int status = launch_raw(
      compact_q, compact_k, boundary_state, v_new, cumulative_gate,
      token_count, raw_output, cuda_stream);
  if (status != static_cast<int>(cudaSuccess)) {
    return status;
  }
  return launch_norm_rows8(
      raw_output, norm_weight, silu_gate,
      token_count * kRowsPerToken, norm_epsilon, output, cuda_stream);
}

int launch_wmma_oracle(const std::uint16_t* const compact_q,
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
      static_cast<unsigned int>((token_count + kChunk - 1U) / kChunk);
  const dim3 grid(2U, chunk_count, kValueHeads);
  (void)cudaGetLastError();
  chunk_o_bv64_wmma_oracle_kernel<<<grid, kChunkThreads, 0U, stream>>>(
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

int launch_fragment_sentinel(
    const std::uint16_t* const matrix_a,
    const std::uint16_t* const matrix_b,
    std::uint32_t* const loaded_a,
    std::uint32_t* const direct_a,
    std::uint32_t* const loaded_b,
    std::uint32_t* const direct_b,
    float* const loaded_accumulator,
    float* const direct_accumulator,
    void* const cuda_stream) noexcept {
  if (matrix_a == nullptr || matrix_b == nullptr || loaded_a == nullptr ||
      direct_a == nullptr || loaded_b == nullptr || direct_b == nullptr ||
      loaded_accumulator == nullptr || direct_accumulator == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  fragment_lane_sentinel_kernel<<<1U, 32U, 0U, stream>>>(
      matrix_a, matrix_b, loaded_a, direct_a, loaded_b, direct_b,
      loaded_accumulator, direct_accumulator);
  return static_cast<int>(cudaGetLastError());
}

int launch_norm_rows8(const std::uint16_t* const raw_output,
                      const std::uint16_t* const norm_weight,
                      const std::uint16_t* const silu_gate,
                      const std::size_t row_count,
                      const float norm_epsilon,
                      std::uint16_t* const output,
                      void* const cuda_stream) noexcept {
  if (raw_output == nullptr || norm_weight == nullptr ||
      silu_gate == nullptr || row_count == 0U ||
      row_count > kMaximumNormTokens * kRowsPerToken ||
      output == nullptr ||
      !std::isfinite(norm_epsilon) || norm_epsilon <= 0.0F) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const unsigned int rows = static_cast<unsigned int>(row_count);
  const unsigned int blocks =
      (rows + kNormRowsPerCta - 1U) / kNormRowsPerCta;
  (void)cudaGetLastError();
  rms_norm_silu_rows8_kernel<<<blocks, kNormThreads, 0U, stream>>>(
      raw_output, norm_weight, silu_gate, rows, norm_epsilon, output);
  return static_cast<int>(cudaGetLastError());
}

int validate_norm_rows8_k256_a4(
    const std::uint16_t* const raw_output_tile,
    const std::uint16_t* const norm_weight,
    const std::uint16_t* const silu_gate_tile,
    const std::size_t tile_logical_token_count,
    const float norm_epsilon,
    const std::size_t destination_first_token,
    const std::size_t whole_logical_token_count,
    const std::size_t launch_token_count,
    const float clip_ratio,
    std::uint8_t* const packed_a_base,
    const std::size_t packed_a_capacity_bytes,
    std::uint16_t* const scales_bf16_base,
    const std::size_t scale_capacity_elements) noexcept {
  const std::size_t expected_launch_token_count =
      q3x::kernels::sm87_a4w4_attention_k256_launch_token_count(
          whole_logical_token_count);
  if (tile_logical_token_count == 0U ||
      tile_logical_token_count > kMaximumTokens ||
      whole_logical_token_count == 0U ||
      whole_logical_token_count > kMaximumNormTokens ||
      expected_launch_token_count == 0U ||
      launch_token_count != expected_launch_token_count ||
      destination_first_token > whole_logical_token_count ||
      tile_logical_token_count >
          whole_logical_token_count - destination_first_token ||
      !std::isfinite(norm_epsilon) || norm_epsilon <= 0.0F ||
      !(clip_ratio > 0.0F && clip_ratio <= 1.0F) ||
      !aligned(raw_output_tile, alignof(std::uint16_t)) ||
      !aligned(norm_weight, alignof(std::uint16_t)) ||
      !aligned(silu_gate_tile, alignof(std::uint16_t)) ||
      !aligned(packed_a_base, 16U) ||
      !aligned(scales_bf16_base, 16U)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const std::size_t required_packed_bytes =
      q3x::kernels::sm87_a4w4_attention_k256_packed_capacity_bytes(
          launch_token_count, kK256A4InputSize);
  const std::size_t required_scale_elements =
      q3x::kernels::sm87_a4w4_attention_k256_scale_capacity_elements(
          launch_token_count, kK256A4InputSize);
  if (required_packed_bytes == 0U || required_scale_elements == 0U ||
      packed_a_capacity_bytes < required_packed_bytes ||
      scale_capacity_elements < required_scale_elements ||
      !q3x::kernels::sm87_a4w4_attention_k256_product_fits(
          required_scale_elements, sizeof(std::uint16_t)) ||
      !q3x::kernels::sm87_a4w4_attention_k256_product_fits(
          tile_logical_token_count, kK256A4InputSize) ||
      !q3x::kernels::sm87_a4w4_attention_k256_product_fits(
          tile_logical_token_count * kK256A4InputSize,
          sizeof(std::uint16_t))) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const std::size_t input_elements =
      tile_logical_token_count * kK256A4InputSize;
  const std::size_t input_bytes = input_elements * sizeof(std::uint16_t);
  const std::size_t scale_bytes =
      required_scale_elements * sizeof(std::uint16_t);
  // raw, gate, packed and scales are independent ownership domains.  Reject
  // every overlap before launching so an invalid slice cannot partially
  // corrupt a span-wide handoff buffer.
  if (byte_ranges_overlap(raw_output_tile, input_bytes,
                          silu_gate_tile, input_bytes) ||
      byte_ranges_overlap(raw_output_tile, input_bytes,
                          packed_a_base, required_packed_bytes) ||
      byte_ranges_overlap(raw_output_tile, input_bytes,
                          scales_bf16_base, scale_bytes) ||
      byte_ranges_overlap(silu_gate_tile, input_bytes,
                          packed_a_base, required_packed_bytes) ||
      byte_ranges_overlap(silu_gate_tile, input_bytes,
                          scales_bf16_base, scale_bytes) ||
      byte_ranges_overlap(packed_a_base, required_packed_bytes,
                          scales_bf16_base, scale_bytes)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const bool final_slice =
      tile_logical_token_count ==
      whole_logical_token_count - destination_first_token;
  const std::size_t padding_tokens =
      final_slice ? launch_token_count - whole_logical_token_count : 0U;
  if (padding_tokens >=
          q3x::kernels::kSm87A4W4AttentionK256TileM ||
      tile_logical_token_count >
          std::numeric_limits<std::size_t>::max() - padding_tokens) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const std::size_t publish_token_count =
      tile_logical_token_count + padding_tokens;
  if (destination_first_token > launch_token_count ||
      publish_token_count > launch_token_count - destination_first_token ||
      publish_token_count == 0U ||
      publish_token_count >
          std::numeric_limits<unsigned int>::max() /
              kHeadOctetsPerToken) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  return static_cast<int>(cudaSuccess);
}

int launch_norm_rows8_k256_a4(
    const std::uint16_t* const raw_output_tile,
    const std::uint16_t* const norm_weight,
    const std::uint16_t* const silu_gate_tile,
    const std::size_t tile_logical_token_count,
    const float norm_epsilon,
    const std::size_t destination_first_token,
    const std::size_t whole_logical_token_count,
    const std::size_t launch_token_count,
    const float clip_ratio,
    std::uint8_t* const packed_a_base,
    const std::size_t packed_a_capacity_bytes,
    std::uint16_t* const scales_bf16_base,
    const std::size_t scale_capacity_elements,
    void* const cuda_stream) noexcept {
  const int validation_status = validate_norm_rows8_k256_a4(
      raw_output_tile, norm_weight, silu_gate_tile,
      tile_logical_token_count, norm_epsilon, destination_first_token,
      whole_logical_token_count, launch_token_count, clip_ratio,
      packed_a_base, packed_a_capacity_bytes, scales_bf16_base,
      scale_capacity_elements);
  if (validation_status != static_cast<int>(cudaSuccess)) {
    return validation_status;
  }
  const bool final_slice =
      tile_logical_token_count ==
      whole_logical_token_count - destination_first_token;
  const std::size_t padding_tokens =
      final_slice ? launch_token_count - whole_logical_token_count : 0U;
  const std::size_t publish_token_count =
      tile_logical_token_count + padding_tokens;
  const unsigned int blocks = static_cast<unsigned int>(
      publish_token_count * kHeadOctetsPerToken);
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  rms_norm_silu_rows8_k256_a4_kernel<<<
      blocks, kNormThreads, 0U, stream>>>(
      raw_output_tile, norm_weight, silu_gate_tile,
      static_cast<unsigned int>(tile_logical_token_count), norm_epsilon,
      static_cast<unsigned int>(destination_first_token),
      static_cast<unsigned int>(publish_token_count), clip_ratio,
      packed_a_base, scales_bf16_base);
  return static_cast<int>(cudaGetLastError());
}

int validate_norm_rows8_factorized_lane_r1_a4(
    const std::uint16_t* const raw_output_tile,
    const std::uint16_t* const norm_weight,
    const std::uint16_t* const silu_gate_tile,
    const std::size_t tile_logical_token_count,
    const float norm_epsilon,
    const float* const authenticated_inverse_alpha_fp32,
    const std::size_t inverse_alpha_capacity_elements,
    const std::size_t destination_first_token,
    const std::size_t whole_logical_token_count,
    const std::size_t launch_token_count,
    const float clip_ratio,
    std::uint8_t* const packed_a_base,
    const std::size_t packed_a_capacity_bytes,
    std::uint16_t* const scales_bf16_base,
    const std::size_t scale_capacity_elements) noexcept {
  const std::size_t expected_launch_token_count =
      q3x::kernels::sm87_a4w4_attention_k256_launch_token_count(
          whole_logical_token_count);
  const auto plan =
      q3x::kernels::sm87_a4w4_factorized_lane_quantize_plan(
          whole_logical_token_count, launch_token_count,
          kK256A4InputSize, kFactorizedLaneR1LaneCount);
  if (tile_logical_token_count == 0U ||
      tile_logical_token_count > kMaximumTokens ||
      whole_logical_token_count == 0U ||
      whole_logical_token_count > kMaximumNormTokens ||
      expected_launch_token_count == 0U ||
      launch_token_count != expected_launch_token_count || !plan.valid() ||
      destination_first_token > whole_logical_token_count ||
      tile_logical_token_count >
          whole_logical_token_count - destination_first_token ||
      inverse_alpha_capacity_elements < kK256A4InputSize ||
      !std::isfinite(norm_epsilon) || norm_epsilon <= 0.0F ||
      !(clip_ratio > 0.0F && clip_ratio <= 1.0F) ||
      !aligned(raw_output_tile, alignof(std::uint16_t)) ||
      !aligned(norm_weight, alignof(std::uint16_t)) ||
      !aligned(silu_gate_tile, alignof(std::uint16_t)) ||
      !aligned(authenticated_inverse_alpha_fp32, 16U) ||
      !aligned(packed_a_base, 16U) ||
      !aligned(scales_bf16_base, 16U) ||
      packed_a_capacity_bytes < plan.packed_capacity_bytes ||
      scale_capacity_elements < plan.scale_capacity_elements) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  if (!q3x::kernels::sm87_a4w4_attention_k256_product_fits(
          tile_logical_token_count, kK256A4InputSize) ||
      !q3x::kernels::sm87_a4w4_attention_k256_product_fits(
          tile_logical_token_count * kK256A4InputSize,
          sizeof(std::uint16_t)) ||
      !q3x::kernels::sm87_a4w4_attention_k256_product_fits(
          kK256A4InputSize, sizeof(float)) ||
      !q3x::kernels::sm87_a4w4_attention_k256_product_fits(
          plan.scale_capacity_elements, sizeof(std::uint16_t))) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const std::size_t input_bytes = tile_logical_token_count *
                                  kK256A4InputSize *
                                  sizeof(std::uint16_t);
  const std::size_t norm_bytes =
      kValueDimension * sizeof(std::uint16_t);
  const std::size_t inverse_bytes = kK256A4InputSize * sizeof(float);
  const std::size_t scale_bytes =
      plan.scale_capacity_elements * sizeof(std::uint16_t);
  const void* const pointers[] = {
      raw_output_tile,
      norm_weight,
      silu_gate_tile,
      authenticated_inverse_alpha_fp32,
      packed_a_base,
      scales_bf16_base};
  const std::size_t bytes[] = {
      input_bytes,
      norm_bytes,
      input_bytes,
      inverse_bytes,
      plan.packed_capacity_bytes,
      scale_bytes};
  for (std::size_t first = 0U; first < 6U; ++first) {
    for (std::size_t second = first + 1U; second < 6U; ++second) {
      if (byte_ranges_overlap(pointers[first], bytes[first],
                              pointers[second], bytes[second])) {
        return static_cast<int>(cudaErrorInvalidValue);
      }
    }
  }

  const bool final_slice =
      tile_logical_token_count ==
      whole_logical_token_count - destination_first_token;
  const std::size_t padding_tokens =
      final_slice ? launch_token_count - whole_logical_token_count : 0U;
  if (padding_tokens >=
          q3x::kernels::kSm87A4W4AttentionK256TileM ||
      tile_logical_token_count >
          std::numeric_limits<std::size_t>::max() - padding_tokens) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const std::size_t publish_token_count =
      tile_logical_token_count + padding_tokens;
  if (destination_first_token > launch_token_count ||
      publish_token_count > launch_token_count - destination_first_token ||
      publish_token_count == 0U ||
      publish_token_count > std::numeric_limits<unsigned int>::max()) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  return static_cast<int>(cudaSuccess);
}

int launch_norm_rows8_factorized_lane_r1_a4(
    const std::uint16_t* const raw_output_tile,
    const std::uint16_t* const norm_weight,
    const std::uint16_t* const silu_gate_tile,
    const std::size_t tile_logical_token_count,
    const float norm_epsilon,
    const float* const authenticated_inverse_alpha_fp32,
    const std::size_t inverse_alpha_capacity_elements,
    const std::size_t destination_first_token,
    const std::size_t whole_logical_token_count,
    const std::size_t launch_token_count,
    const float clip_ratio,
    std::uint8_t* const packed_a_base,
    const std::size_t packed_a_capacity_bytes,
    std::uint16_t* const scales_bf16_base,
    const std::size_t scale_capacity_elements,
    void* const cuda_stream) noexcept {
  const int validation_status =
      validate_norm_rows8_factorized_lane_r1_a4(
          raw_output_tile, norm_weight, silu_gate_tile,
          tile_logical_token_count, norm_epsilon,
          authenticated_inverse_alpha_fp32,
          inverse_alpha_capacity_elements, destination_first_token,
          whole_logical_token_count, launch_token_count, clip_ratio,
          packed_a_base, packed_a_capacity_bytes, scales_bf16_base,
          scale_capacity_elements);
  if (validation_status != static_cast<int>(cudaSuccess)) {
    return validation_status;
  }
  const bool final_slice =
      tile_logical_token_count ==
      whole_logical_token_count - destination_first_token;
  const std::size_t padding_tokens =
      final_slice ? launch_token_count - whole_logical_token_count : 0U;
  const std::size_t publish_token_count =
      tile_logical_token_count + padding_tokens;
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  rms_norm_silu_factorized_lane_r1_a4_kernel<<<
      static_cast<unsigned int>(publish_token_count), kNormThreads, 0U,
      stream>>>(
      raw_output_tile, norm_weight, silu_gate_tile,
      static_cast<unsigned int>(tile_logical_token_count), norm_epsilon,
      authenticated_inverse_alpha_fp32,
      static_cast<unsigned int>(destination_first_token),
      static_cast<unsigned int>(publish_token_count), clip_ratio,
      packed_a_base, scales_bf16_base);
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

int query_norm_k256_a4_resources(
    int* const registers_per_thread,
    std::size_t* const static_shared_bytes,
    std::size_t* const local_bytes,
    int* const maximum_threads_per_block,
    int* const active_blocks_per_sm) noexcept {
  const int status = query_kernel_resources(
      rms_norm_silu_rows8_k256_a4_kernel,
      static_cast<int>(kNormThreads), registers_per_thread,
      static_shared_bytes, local_bytes, maximum_threads_per_block,
      active_blocks_per_sm);
  if (status != static_cast<int>(cudaSuccess)) {
    return status;
  }
  if (*registers_per_thread <= 0 ||
      *registers_per_thread >
          static_cast<int>(kMaximumNormK256A4Registers) ||
      *static_shared_bytes != kNormK256A4SharedBytes ||
      *local_bytes != 0U ||
      *maximum_threads_per_block < static_cast<int>(kNormThreads) ||
      *active_blocks_per_sm <
          static_cast<int>(kMinimumNormK256A4CtasPerSm)) {
    return static_cast<int>(cudaErrorLaunchOutOfResources);
  }
  return static_cast<int>(cudaSuccess);
}

int query_norm_factorized_lane_r1_a4_resources(
    int* const registers_per_thread,
    std::size_t* const static_shared_bytes,
    std::size_t* const local_bytes,
    int* const maximum_threads_per_block,
    int* const active_blocks_per_sm) noexcept {
  const int status = query_kernel_resources(
      rms_norm_silu_factorized_lane_r1_a4_kernel,
      static_cast<int>(kNormThreads), registers_per_thread,
      static_shared_bytes, local_bytes, maximum_threads_per_block,
      active_blocks_per_sm);
  if (status != static_cast<int>(cudaSuccess)) {
    return status;
  }
  if (*registers_per_thread <= 0 ||
      *registers_per_thread >
          static_cast<int>(kMaximumNormFactorizedLaneR1A4Registers) ||
      *static_shared_bytes != kNormFactorizedLaneR1A4SharedBytes ||
      *local_bytes != 0U ||
      *maximum_threads_per_block < static_cast<int>(kNormThreads) ||
      *active_blocks_per_sm <
          static_cast<int>(kMinimumNormFactorizedLaneR1A4CtasPerSm)) {
    return static_cast<int>(cudaErrorLaunchOutOfResources);
  }
  return static_cast<int>(cudaSuccess);
}

}  // namespace q3x::runtime::gdn_prefill_chunk_o_bv64_detail
