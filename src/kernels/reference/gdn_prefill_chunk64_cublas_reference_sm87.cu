#include "gdn_prefill_chunk64_cublas_reference_sm87.h"
#include "gdn_prefill_whole_span_conv_sm87.h"
#include "../sm87/gdn_prefill_chunk64_native_sm87.h"
#include "../sm87/gdn_prefill_chunk_o_bv64_sm87.h"
#include "../sm87/gdn_prefill_group_wy_sm87.h"
#include "../sm87/gdn_prefill_wy_vllm_layout_sm87.h"

#include "q3x/runtime/gdn_decode.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>

namespace q3x::runtime::gdn_prefill_chunk64_native_detail {
namespace {

thread_local InspectionHook g_inspection_hook{};
thread_local PreprocessInspectionHook g_preprocess_inspection_hook{};
thread_local WyTimingHook g_wy_timing_hook{};
thread_local bool g_force_fused_kkt_baseline_for_test = false;
thread_local bool g_force_split_wy_baseline_for_test = false;
thread_local bool g_force_packed_qkv_baseline_for_test = false;
thread_local bool g_force_resident_state_baseline_for_test = false;
thread_local VllmLayoutWyRouteForTest
    g_vllm_layout_wy_route_for_test =
        VllmLayoutWyRouteForTest::kProductionDefault;
thread_local bool g_force_legacy_qk_reconstruct_baseline_for_test = false;
thread_local bool g_force_packless_resident_state_fallback_for_test = false;

}  // namespace

InspectionHook exchange_inspection_hook(const InspectionHook hook) noexcept {
  const InspectionHook previous = g_inspection_hook;
  g_inspection_hook = hook;
  return previous;
}

PreprocessInspectionHook exchange_preprocess_inspection_hook(
    const PreprocessInspectionHook hook) noexcept {
  const PreprocessInspectionHook previous = g_preprocess_inspection_hook;
  g_preprocess_inspection_hook = hook;
  return previous;
}

WyTimingHook exchange_wy_timing_hook(const WyTimingHook hook) noexcept {
  const WyTimingHook previous = g_wy_timing_hook;
  g_wy_timing_hook = hook;
  return previous;
}

bool exchange_force_fused_kkt_baseline_for_test(
    const bool enabled) noexcept {
  const bool previous = g_force_fused_kkt_baseline_for_test;
  g_force_fused_kkt_baseline_for_test = enabled;
  return previous;
}

bool exchange_force_split_wy_baseline_for_test(
    const bool enabled) noexcept {
  const bool previous = g_force_split_wy_baseline_for_test;
  g_force_split_wy_baseline_for_test = enabled;
  return previous;
}

bool exchange_force_packed_qkv_baseline_for_test(
    const bool enabled) noexcept {
  const bool previous = g_force_packed_qkv_baseline_for_test;
  g_force_packed_qkv_baseline_for_test = enabled;
  return previous;
}

bool exchange_force_resident_state_baseline_for_test(
    const bool enabled) noexcept {
  const bool previous = g_force_resident_state_baseline_for_test;
  g_force_resident_state_baseline_for_test = enabled;
  return previous;
}

VllmLayoutWyRouteForTest exchange_vllm_layout_wy_route_for_test(
    const VllmLayoutWyRouteForTest route) noexcept {
  const VllmLayoutWyRouteForTest previous =
      g_vllm_layout_wy_route_for_test;
  g_vllm_layout_wy_route_for_test = route;
  return previous;
}

bool exchange_force_legacy_qk_reconstruct_baseline_for_test(
  const bool enabled) noexcept {
  const bool previous = g_force_legacy_qk_reconstruct_baseline_for_test;
  g_force_legacy_qk_reconstruct_baseline_for_test = enabled;
  return previous;
}

bool exchange_force_packless_resident_state_fallback_for_test(
    const bool enabled) noexcept {
  const bool previous =
      g_force_packless_resident_state_fallback_for_test;
  g_force_packless_resident_state_fallback_for_test = enabled;
  return previous;
}

void inspect_native_boundaries(
    const std::uint16_t* const transform,
    const std::size_t transform_elements,
    const std::uint16_t* const w,
    const std::size_t w_elements,
    const std::uint16_t* const u,
    const std::size_t u_elements,
    const std::uint16_t* const v_new,
    const std::size_t v_new_elements,
    const std::uint16_t* const boundary_state,
    const std::size_t boundary_state_elements,
    const std::uint16_t* const compact_k,
    const std::size_t compact_k_elements,
    const float* const gamma,
    const std::size_t gamma_elements,
    const std::uint16_t* const diagnostic_k_decay,
    const std::size_t diagnostic_k_decay_elements,
    const std::uint16_t* const diagnostic_post_update_state,
    const std::size_t diagnostic_post_update_state_elements,
    const std::uint16_t* const state_output,
    const std::size_t state_elements,
    const std::uint16_t* const output,
    const std::size_t output_elements,
    void* const cuda_stream) noexcept {
  const InspectionHook hook = g_inspection_hook;
  if (hook.callback != nullptr) {
    hook.callback(transform, transform_elements, w, w_elements, u,
                  u_elements, v_new, v_new_elements, boundary_state,
                  boundary_state_elements, compact_k, compact_k_elements,
                  gamma, gamma_elements, diagnostic_k_decay,
                  diagnostic_k_decay_elements,
                  diagnostic_post_update_state,
                  diagnostic_post_update_state_elements, state_output,
                  state_elements, output, output_elements, cuda_stream,
                  hook.context);
  }
}

void inspect_preprocess_boundaries(
    const std::uint16_t* const compact_q,
    const std::size_t compact_q_elements,
    const std::uint16_t* const compact_k,
    const std::size_t compact_k_elements,
    void* const cuda_stream) noexcept {
  const PreprocessInspectionHook hook = g_preprocess_inspection_hook;
  if (hook.callback != nullptr) {
    hook.callback(compact_q, compact_q_elements, compact_k,
                  compact_k_elements, cuda_stream, hook.context);
  }
}

}  // namespace q3x::runtime::gdn_prefill_chunk64_native_detail

namespace q3x::runtime::gdn_prefill_chunk64_reference_detail {
namespace {

constexpr unsigned int kThreads = 256U;
constexpr unsigned int kNormalizeThreads = 128U;
constexpr unsigned int kChunkThreads = 64U;
constexpr unsigned int kFusedSolveThreads = 128U;
constexpr unsigned int kFusedSolveWarps = kFusedSolveThreads / 32U;
constexpr unsigned int kSolveSubblock = 16U;
constexpr std::size_t kTokenCount = 512U;
constexpr std::size_t kChunkSize = 64U;
constexpr std::size_t kChunkCount = kTokenCount / kChunkSize;
constexpr std::size_t kQkHeadCount = kGdnQkHeadCount;
constexpr std::size_t kValueHeadCount = kGdnValueHeadCount;
constexpr std::size_t kDimension = kGdnHeadDimension;
constexpr std::size_t kMatrixCount = kChunkCount * kValueHeadCount;
constexpr std::size_t kCompactMatrixCount =
    kChunkCount * kQkHeadCount;
constexpr std::size_t kHeadTokenElements =
    kMatrixCount * kChunkSize * kDimension;
constexpr std::size_t kCompactHeadTokenElements =
    kCompactMatrixCount * kChunkSize * kDimension;
constexpr std::size_t kChunkMatrixElements =
    kMatrixCount * kChunkSize * kChunkSize;
constexpr std::size_t kBoundaryStateElements =
    kMatrixCount * kDimension * kDimension;
constexpr std::size_t kScalarElements = kMatrixCount * kChunkSize;
constexpr std::size_t kWorkspaceAlignment = 256U;
constexpr std::size_t kBf16Bytes = sizeof(std::uint16_t);
constexpr std::size_t kFp32Bytes = sizeof(float);
constexpr std::size_t kQOffset = 0U;
constexpr std::size_t kKOffset = kGdnQElements;
constexpr std::size_t kVOffset = kGdnQElements + kGdnKElements;

static_assert(kTokenCount % kChunkSize == 0U);
static_assert(kQkHeadCount * 3U == kValueHeadCount);
static_assert(kCompactHeadTokenElements * 3U == kHeadTokenElements);
static_assert(kDimension == 128U);
static_assert(kChunkSize == 4U * kSolveSubblock);

namespace wmma = nvcuda::wmma;
using Bf16 = __nv_bfloat16;
using WmmaAccumulator =
    wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

[[nodiscard]] constexpr std::size_t align_workspace(
    const std::size_t value) noexcept {
  return (value + kWorkspaceAlignment - 1U) &
         ~(kWorkspaceAlignment - 1U);
}

[[nodiscard]] constexpr std::size_t append_region(
    const std::size_t offset,
    const std::size_t elements,
    const std::size_t element_bytes) noexcept {
  return align_workspace(offset) + elements * element_bytes;
}

[[nodiscard]] constexpr std::size_t required_workspace_bytes() noexcept {
  std::size_t offset = 0U;
  // Same-ELF packed-baseline capacity is retained for Q, K, exp(g)K,
  // end-decayed K, and V. The production path uses only the compact H16
  // prefix of Q/K and never touches the three packed materializations.
  // T, QK, W, U, Vnew, and H retain their value-head layout.
  for (unsigned int index = 0U; index < 5U; ++index) {
    offset = append_region(offset, kHeadTokenElements, kBf16Bytes);
  }
  for (unsigned int index = 0U; index < 2U; ++index) {
    offset = append_region(offset, kChunkMatrixElements, kBf16Bytes);
  }
  for (unsigned int index = 0U; index < 3U; ++index) {
    offset = append_region(offset, kHeadTokenElements, kBf16Bytes);
  }
  offset = append_region(offset, kBoundaryStateElements, kBf16Bytes);
  // The open-book KKT path deliberately preserves FLA's FP32 A boundary
  // between the streamed dot product and the triangular inverse. This costs
  // one bounded C64 matrix per chunk/head and lets the two kernels use 8 KiB
  // and 32 KiB of shared memory instead of sharing a 48 KiB fused lifetime.
  offset = append_region(offset, kChunkMatrixElements, kFp32Bytes);
  // Cumulative gate and beta. QK is produced directly at its BF16 stage
  // boundary by the native WMMA kernel.
  offset = append_region(offset, kScalarElements, kFp32Bytes);
  offset = append_region(offset, kScalarElements, kFp32Bytes);
  return align_workspace(offset);
}

struct Workspace {
  std::uint16_t* q = nullptr;
  std::uint16_t* k = nullptr;
  std::uint16_t* k_g = nullptr;
  std::uint16_t* k_decay = nullptr;
  std::uint16_t* v = nullptr;
  std::uint16_t* transform = nullptr;
  std::uint16_t* qk = nullptr;
  std::uint16_t* w = nullptr;
  std::uint16_t* u = nullptr;
  std::uint16_t* v_new = nullptr;
  std::uint16_t* boundary_state = nullptr;
  float* kkt = nullptr;
  float* gamma = nullptr;
  float* beta = nullptr;
};

template <typename T>
[[nodiscard]] T* take_region(std::uint8_t* const base,
                             std::size_t& offset,
                             const std::size_t elements) noexcept {
  offset = align_workspace(offset);
  auto* const result = reinterpret_cast<T*>(base + offset);
  offset += elements * sizeof(T);
  return result;
}

[[nodiscard]] bool partition_workspace(void* const raw,
                                       const std::size_t capacity,
                                       Workspace& workspace) noexcept {
  if (raw == nullptr || capacity < required_workspace_bytes()) {
    return false;
  }
  auto* const base = static_cast<std::uint8_t*>(raw);
  std::size_t offset = 0U;
  workspace.q = take_region<std::uint16_t>(base, offset, kHeadTokenElements);
  workspace.k = take_region<std::uint16_t>(base, offset, kHeadTokenElements);
  workspace.k_g =
      take_region<std::uint16_t>(base, offset, kHeadTokenElements);
  workspace.k_decay =
      take_region<std::uint16_t>(base, offset, kHeadTokenElements);
  workspace.v = take_region<std::uint16_t>(base, offset, kHeadTokenElements);
  workspace.transform =
      take_region<std::uint16_t>(base, offset, kChunkMatrixElements);
  workspace.qk =
      take_region<std::uint16_t>(base, offset, kChunkMatrixElements);
  workspace.w = take_region<std::uint16_t>(base, offset, kHeadTokenElements);
  workspace.u = take_region<std::uint16_t>(base, offset, kHeadTokenElements);
  workspace.v_new =
      take_region<std::uint16_t>(base, offset, kHeadTokenElements);
  workspace.boundary_state =
      take_region<std::uint16_t>(base, offset, kBoundaryStateElements);
  workspace.kkt = take_region<float>(base, offset, kChunkMatrixElements);
  workspace.gamma = take_region<float>(base, offset, kScalarElements);
  workspace.beta = take_region<float>(base, offset, kScalarElements);
  return align_workspace(offset) <= capacity;
}

__device__ __forceinline__ float decode_bf16_device(
    const std::uint16_t value) {
  return __uint_as_float(static_cast<unsigned int>(value) << 16U);
}

__device__ __forceinline__ std::uint16_t encode_bf16_device(
    const float value) {
  const unsigned int bits = __float_as_uint(value);
  if ((bits & 0x7fffffffU) > 0x7f800000U) {
    return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
  }
  return __bfloat16_as_ushort(__float2bfloat16_rn(value));
}

__device__ __forceinline__ float stable_softplus_device(const float value) {
  return value > 20.0F ? value : log1pf(expf(value));
}

__device__ __forceinline__ float stable_sigmoid_device(const float value) {
  if (value >= 0.0F) {
    return 1.0F / (1.0F + expf(-value));
  }
  const float exponential = expf(value);
  return exponential / (1.0F + exponential);
}

template <bool Compact>
__global__ __launch_bounds__(kNormalizeThreads) void normalize_qk_kernel(
    const std::uint16_t* const conv_qkv,
    const float l2_epsilon,
    std::uint16_t* const q,
    std::uint16_t* const k) {
  __shared__ float q_squares[kDimension];
  __shared__ float k_squares[kDimension];
  const unsigned int dimension = threadIdx.x;
  const std::size_t token = blockIdx.x / kQkHeadCount;
  const std::size_t qk_head = blockIdx.x % kQkHeadCount;
  const std::size_t source =
      token * kGdnQkvChannels + qk_head * kDimension + dimension;
  const float q_value = decode_bf16_device(conv_qkv[kQOffset + source]);
  const float k_value = decode_bf16_device(conv_qkv[kKOffset + source]);
  q_squares[dimension] = q_value * q_value;
  k_squares[dimension] = k_value * k_value;
  __syncthreads();
  for (unsigned int stride = kNormalizeThreads / 2U; stride != 0U;
       stride >>= 1U) {
    if (dimension < stride) {
      q_squares[dimension] += q_squares[dimension + stride];
      k_squares[dimension] += k_squares[dimension + stride];
    }
    __syncthreads();
  }
  const float q_scale =
      rsqrtf(q_squares[0] + l2_epsilon) *
      rsqrtf(static_cast<float>(kDimension));
  const float k_scale = rsqrtf(k_squares[0] + l2_epsilon);
  const std::size_t chunk = token / kChunkSize;
  const std::size_t token_in_chunk = token % kChunkSize;
  if constexpr (Compact) {
    const std::size_t matrix = chunk * kQkHeadCount + qk_head;
    const std::size_t destination =
        (matrix * kChunkSize + token_in_chunk) * kDimension + dimension;
    q[destination] = encode_bf16_device(q_value * q_scale);
    k[destination] = encode_bf16_device(k_value * k_scale);
  } else {
#pragma unroll
    for (std::size_t replica = 0U; replica < 3U; ++replica) {
      const std::size_t value_head = qk_head * 3U + replica;
      const std::size_t matrix = chunk * kValueHeadCount + value_head;
      const std::size_t destination =
          (matrix * kChunkSize + token_in_chunk) * kDimension + dimension;
      q[destination] = encode_bf16_device(q_value * q_scale);
      k[destination] = encode_bf16_device(k_value * k_scale);
    }
  }
}

__global__ __launch_bounds__(kChunkThreads) void prepare_gate_kernel(
    const std::uint16_t* const a,
    const std::uint16_t* const b,
    const std::uint16_t* const A_log,
    const std::uint16_t* const dt_bias,
    float* const gamma,
    float* const beta) {
  __shared__ float log_alpha[kChunkSize];
  const unsigned int token_in_chunk = threadIdx.x;
  const std::size_t matrix = blockIdx.x;
  const std::size_t chunk = matrix / kValueHeadCount;
  const std::size_t value_head = matrix % kValueHeadCount;
  const std::size_t token = chunk * kChunkSize + token_in_chunk;
  const std::size_t scalar = token * kValueHeadCount + value_head;
  const float gate_input = decode_bf16_device(a[scalar]) +
                           decode_bf16_device(dt_bias[value_head]);
  const float g = -expf(decode_bf16_device(A_log[value_head])) *
                  stable_softplus_device(gate_input);
  log_alpha[token_in_chunk] = g;
  beta[matrix * kChunkSize + token_in_chunk] =
      stable_sigmoid_device(decode_bf16_device(b[scalar]));
  __syncthreads();
  if (token_in_chunk == 0U) {
    float cumulative = 0.0F;
#pragma unroll
    for (unsigned int index = 0U; index < kChunkSize; ++index) {
      cumulative += log_alpha[index];
      gamma[matrix * kChunkSize + index] = cumulative;
    }
  }
}

__global__ void pack_scaled_k_v_kernel(
    const std::uint16_t* const conv_qkv,
    const std::uint16_t* const k,
    const float* const gamma,
    const std::size_t element_count,
    std::uint16_t* const k_g,
    std::uint16_t* const k_decay,
    std::uint16_t* const v) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= element_count) {
    return;
  }
  const std::size_t dimension = index % kDimension;
  const std::size_t token_matrix_index = index / kDimension;
  const std::size_t token_in_chunk = token_matrix_index % kChunkSize;
  const std::size_t matrix = token_matrix_index / kChunkSize;
  const std::size_t chunk = matrix / kValueHeadCount;
  const std::size_t value_head = matrix % kValueHeadCount;
  const std::size_t token = chunk * kChunkSize + token_in_chunk;
  const float cumulative = gamma[matrix * kChunkSize + token_in_chunk];
  const float final_cumulative =
      gamma[matrix * kChunkSize + kChunkSize - 1U];
  const float k_value = decode_bf16_device(k[index]);
  k_g[index] = encode_bf16_device(expf(cumulative) * k_value);
  k_decay[index] =
      encode_bf16_device(expf(final_cumulative - cumulative) * k_value);
  const std::size_t source =
      token * kGdnQkvChannels + kVOffset +
      value_head * kDimension + dimension;
  v[index] = conv_qkv[source];
}

__device__ __forceinline__ void lower_subblock_coordinates(
    const unsigned int block,
    unsigned int& row_block,
    unsigned int& column_block) {
  if (block == 0U) {
    row_block = 0U;
    column_block = 0U;
  } else if (block <= 2U) {
    row_block = 1U;
    column_block = block - 1U;
  } else if (block <= 5U) {
    row_block = 2U;
    column_block = block - 3U;
  } else {
    row_block = 3U;
    column_block = block - 6U;
  }
}

__device__ __forceinline__ void warp_accumulate_row_major_product(
    WmmaAccumulator& accumulator,
    const Bf16* const a,
    const int leading_a,
    const Bf16* const b,
    const int leading_b) {
  wmma::fragment<wmma::matrix_a, 16, 16, 16, Bf16, wmma::row_major>
      a_fragment;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, Bf16, wmma::row_major>
      b_fragment;
  wmma::load_matrix_sync(a_fragment, a, leading_a);
  wmma::load_matrix_sync(b_fragment, b, leading_b);
  wmma::mma_sync(accumulator, a_fragment, b_fragment, accumulator);
}

__device__ __forceinline__ void warp_store_and_encode_block(
    WmmaAccumulator& accumulator,
    float* const output,
    Bf16* const encoded_output,
    const bool negate) {
  const unsigned int lane = threadIdx.x % 32U;
  wmma::store_matrix_sync(output, accumulator,
                          static_cast<int>(kChunkSize),
                          wmma::mem_row_major);
  __syncwarp();
  for (unsigned int index = lane;
       index < kSolveSubblock * kSolveSubblock; index += 32U) {
    const unsigned int row = index / kSolveSubblock;
    const unsigned int column = index % kSolveSubblock;
    float value = output[row * kChunkSize + column];
    if (negate) {
      value = -value;
      output[row * kChunkSize + column] = value;
    }
    encoded_output[row * kChunkSize + column] =
        __float2bfloat16_rn(value);
  }
  __syncwarp();
}

__device__ __forceinline__ void warp_form_inverse_block(
    const Bf16* const diagonal_inverse,
    const Bf16* const left0,
    const Bf16* const right0,
    const Bf16* const left1,
    const Bf16* const right1,
    const Bf16* const left2,
    const Bf16* const right2,
    float* const output,
    Bf16* const encoded_output) {
  WmmaAccumulator inner;
  wmma::fill_fragment(inner, 0.0F);
  warp_accumulate_row_major_product(inner, left0,
                                    static_cast<int>(kChunkSize), right0,
                                    static_cast<int>(kChunkSize));
  if (left1 != nullptr && right1 != nullptr) {
    warp_accumulate_row_major_product(inner, left1,
                                      static_cast<int>(kChunkSize), right1,
                                      static_cast<int>(kChunkSize));
  }
  if (left2 != nullptr && right2 != nullptr) {
    warp_accumulate_row_major_product(inner, left2,
                                      static_cast<int>(kChunkSize), right2,
                                      static_cast<int>(kChunkSize));
  }
  warp_store_and_encode_block(inner, output, encoded_output, false);

  WmmaAccumulator outer;
  wmma::fill_fragment(outer, 0.0F);
  warp_accumulate_row_major_product(
      outer, diagonal_inverse, static_cast<int>(kChunkSize), encoded_output,
      static_cast<int>(kChunkSize));
  warp_store_and_encode_block(outer, output, encoded_output, true);
}

// Architecture stage A: fuse K K^T, gate/beta application, and the C64
// triangular solve. The scalar reference above performs a full 64-step solve
// per matrix. This route instead solves four independent 16x16 diagonal
// blocks and uses BF16 Tensor-Core block products to construct the six lower
// inverse blocks. It deliberately preserves a BF16 transform boundary for
// the later W/U stages.
__global__ __launch_bounds__(kFusedSolveThreads)
void fused_kkt_solve_block16_kernel(
    const std::uint16_t* const k,
    const float* const gamma,
    const float* const beta,
    std::uint16_t* const transform) {
  extern __shared__ unsigned char shared_raw[];
  auto* const shared_k = reinterpret_cast<Bf16*>(shared_raw);
  auto* const shared_l = reinterpret_cast<float*>(
      shared_raw + kChunkSize * kDimension * sizeof(Bf16));
  auto* const shared_inverse = shared_l + kChunkSize * kChunkSize;
  auto* const shared_l_bf16 = shared_k;
  auto* const shared_inverse_bf16 =
      shared_k + kChunkSize * kChunkSize;

  const unsigned int thread = threadIdx.x;
  const unsigned int warp = thread / 32U;
  const unsigned int lane = thread % 32U;
  const std::size_t matrix = blockIdx.x;
  const auto* const matrix_k = reinterpret_cast<const Bf16*>(
      k + matrix * kChunkSize * kDimension);
  const float* const matrix_gamma = gamma + matrix * kChunkSize;
  const float* const matrix_beta = beta + matrix * kChunkSize;

  for (unsigned int index = thread;
       index < kChunkSize * kDimension; index += kFusedSolveThreads) {
    shared_k[index] = matrix_k[index];
  }
  for (unsigned int index = thread;
       index < kChunkSize * kChunkSize; index += kFusedSolveThreads) {
    shared_l[index] = 0.0F;
    shared_inverse[index] = 0.0F;
  }
  __syncthreads();

  for (unsigned int lower_block = warp; lower_block < 10U;
       lower_block += kFusedSolveWarps) {
    unsigned int row_block = 0U;
    unsigned int column_block = 0U;
    lower_subblock_coordinates(lower_block, row_block, column_block);
    WmmaAccumulator accumulator;
    wmma::fill_fragment(accumulator, 0.0F);
    for (unsigned int dimension_block = 0U;
         dimension_block < kDimension / kSolveSubblock;
         ++dimension_block) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, Bf16,
                     wmma::row_major>
          a_fragment;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, Bf16,
                     wmma::col_major>
          b_fragment;
      const Bf16* const a =
          shared_k + row_block * kSolveSubblock * kDimension +
          dimension_block * kSolveSubblock;
      const Bf16* const b =
          shared_k + column_block * kSolveSubblock * kDimension +
          dimension_block * kSolveSubblock;
      wmma::load_matrix_sync(a_fragment, a,
                             static_cast<int>(kDimension));
      wmma::load_matrix_sync(b_fragment, b,
                             static_cast<int>(kDimension));
      wmma::mma_sync(accumulator, a_fragment, b_fragment, accumulator);
    }
    float* const output =
        shared_l + row_block * kSolveSubblock * kChunkSize +
        column_block * kSolveSubblock;
    wmma::store_matrix_sync(output, accumulator,
                            static_cast<int>(kChunkSize),
                            wmma::mem_row_major);
  }
  __syncthreads();

  for (unsigned int index = thread;
       index < kChunkSize * kChunkSize; index += kFusedSolveThreads) {
    const unsigned int row = index / kChunkSize;
    const unsigned int column = index % kChunkSize;
    float value = 0.0F;
    if (row == column) {
      value = 1.0F;
    } else if (row > column) {
      value = matrix_beta[row] *
              expf(matrix_gamma[row] - matrix_gamma[column]) *
              shared_l[index];
    }
    shared_l[index] = value;
  }
  __syncthreads();

  // Four warps independently invert the four 16x16 diagonal blocks. Lanes
  // 0..15 own one inverse column and synchronize after each solved row.
  const unsigned int diagonal_base =
      warp * kSolveSubblock * kChunkSize + warp * kSolveSubblock;
  for (unsigned int row = 0U; row < kSolveSubblock; ++row) {
    if (lane < kSolveSubblock) {
      float value = row == lane ? 1.0F : 0.0F;
      if (lane <= row) {
        for (unsigned int inner = lane; inner < row; ++inner) {
          value -=
              shared_l[diagonal_base + row * kChunkSize + inner] *
              shared_inverse[diagonal_base +
                             inner * kChunkSize + lane];
        }
      } else {
        value = 0.0F;
      }
      shared_inverse[diagonal_base + row * kChunkSize + lane] = value;
    }
    __syncwarp();
  }
  __syncthreads();

  // K is dead after KKT. Reuse its 16 KiB bank for BF16 L and inverse
  // mirrors consumed by the WMMA block merge.
  for (unsigned int index = thread;
       index < kChunkSize * kChunkSize; index += kFusedSolveThreads) {
    shared_l_bf16[index] = __float2bfloat16_rn(shared_l[index]);
    shared_inverse_bf16[index] =
        __float2bfloat16_rn(shared_inverse[index]);
  }
  __syncthreads();

  // First subdiagonal: inv(i,j) = -inv(i,i) L(i,j) inv(j,j).
  if (warp < 3U) {
    const unsigned int row_block = warp + 1U;
    const unsigned int column_block = warp;
    const unsigned int output_base =
        row_block * kSolveSubblock * kChunkSize +
        column_block * kSolveSubblock;
    warp_form_inverse_block(
        shared_inverse_bf16 +
            row_block * kSolveSubblock * kChunkSize +
            row_block * kSolveSubblock,
        shared_l_bf16 + output_base,
        shared_inverse_bf16 +
            column_block * kSolveSubblock * kChunkSize +
            column_block * kSolveSubblock,
        nullptr, nullptr, nullptr, nullptr,
        shared_inverse + output_base,
        shared_inverse_bf16 + output_base);
  }
  __syncthreads();

  // Second subdiagonal. Each inner accumulator combines both block paths
  // before the left diagonal inverse is applied.
  if (warp < 2U) {
    const unsigned int row_block = warp + 2U;
    const unsigned int column_block = warp;
    const unsigned int middle_block = warp + 1U;
    const unsigned int output_base =
        row_block * kSolveSubblock * kChunkSize +
        column_block * kSolveSubblock;
    warp_form_inverse_block(
        shared_inverse_bf16 +
            row_block * kSolveSubblock * kChunkSize +
            row_block * kSolveSubblock,
        shared_l_bf16 + output_base,
        shared_inverse_bf16 +
            column_block * kSolveSubblock * kChunkSize +
            column_block * kSolveSubblock,
        shared_l_bf16 +
            row_block * kSolveSubblock * kChunkSize +
            middle_block * kSolveSubblock,
        shared_inverse_bf16 +
            middle_block * kSolveSubblock * kChunkSize +
            column_block * kSolveSubblock,
        nullptr, nullptr, shared_inverse + output_base,
        shared_inverse_bf16 + output_base);
  }
  __syncthreads();

  // Bottom-left block has three paths through the lower block triangle.
  if (warp == 0U) {
    constexpr unsigned int row_block = 3U;
    constexpr unsigned int output_base =
        row_block * kSolveSubblock * kChunkSize;
    warp_form_inverse_block(
        shared_inverse_bf16 +
            row_block * kSolveSubblock * kChunkSize +
            row_block * kSolveSubblock,
        shared_l_bf16 + output_base,
        shared_inverse_bf16,
        shared_l_bf16 + output_base + kSolveSubblock,
        shared_inverse_bf16 + kSolveSubblock * kChunkSize,
        shared_l_bf16 + output_base + 2U * kSolveSubblock,
        shared_inverse_bf16 + 2U * kSolveSubblock * kChunkSize,
        shared_inverse + output_base,
        shared_inverse_bf16 + output_base);
  }
  __syncthreads();

  auto* const matrix_transform = reinterpret_cast<Bf16*>(
      transform + matrix * kChunkSize * kChunkSize);
  for (unsigned int index = thread;
       index < kChunkSize * kChunkSize; index += kFusedSolveThreads) {
    const unsigned int row = index / kChunkSize;
    const unsigned int column = index % kChunkSize;
    matrix_transform[row + column * kChunkSize] =
        __float2bfloat16_rn(shared_inverse[index] * matrix_beta[column]);
  }
}

// P513's exact vLLM/FLA specialization is C64 x K128 with BK64, four warps,
// and 8 KiB of dynamic shared memory. The important property is the dataflow,
// not Triton's launch syntax: one warp owns a 16-row strip and keeps all four
// output fragments live while two K64 panels pass through shared memory.
// Every K element is therefore fetched once per CTA instead of occupying a
// full K128 shared lifetime beside the triangular inverse.
constexpr unsigned int kStreamedKktThreads = 128U;
constexpr unsigned int kStreamedKktWarps = kStreamedKktThreads / 32U;
constexpr unsigned int kStreamedKktPanel = 64U;
constexpr unsigned int kStreamedKktVectorsPerRow =
    kStreamedKktPanel * sizeof(Bf16) / sizeof(uint4);
constexpr unsigned int kStreamedKktVectorCount =
    kChunkSize * kStreamedKktVectorsPerRow;
constexpr std::size_t kStreamedKktSharedBytes =
    kChunkSize * kStreamedKktPanel * sizeof(Bf16);
constexpr unsigned int kStreamedKktScratchFloats =
    kStreamedKktWarps * kSolveSubblock * kSolveSubblock;

static_assert(kStreamedKktWarps == 4U);
static_assert(kStreamedKktPanel == 4U * kSolveSubblock);
static_assert(kStreamedKktSharedBytes >=
              (kStreamedKktScratchFloats + 2U * kChunkSize) *
                  sizeof(float));

__global__ __launch_bounds__(kStreamedKktThreads)
void streamed_kkt_gate_chunk64_kernel(
    const std::uint16_t* const k,
    const float* const gamma,
    const float* const beta,
    float* const kkt) {
  extern __shared__ unsigned char shared_raw[];
  auto* const shared_k = reinterpret_cast<Bf16*>(shared_raw);
  const unsigned int thread = threadIdx.x;
  const unsigned int warp = thread / 32U;
  const unsigned int lane = thread % 32U;
  const std::size_t matrix = blockIdx.x;
  const auto* const matrix_k = reinterpret_cast<const Bf16*>(
      k + matrix * kChunkSize * kDimension);
  const float* const matrix_gamma = gamma + matrix * kChunkSize;
  const float* const matrix_beta = beta + matrix * kChunkSize;
  float* const matrix_kkt = kkt + matrix * kChunkSize * kChunkSize;

  WmmaAccumulator accumulators[4];
#pragma unroll
  for (unsigned int column_block = 0U; column_block < 4U;
       ++column_block) {
    wmma::fill_fragment(accumulators[column_block], 0.0F);
  }

#pragma unroll
  for (unsigned int panel = 0U; panel < kDimension / kStreamedKktPanel;
       ++panel) {
    for (unsigned int vector = thread; vector < kStreamedKktVectorCount;
         vector += kStreamedKktThreads) {
      const unsigned int row = vector / kStreamedKktVectorsPerRow;
      const unsigned int vector_in_row =
          vector % kStreamedKktVectorsPerRow;
      const auto* const source = reinterpret_cast<const uint4*>(
          matrix_k + row * kDimension + panel * kStreamedKktPanel) +
          vector_in_row;
      reinterpret_cast<uint4*>(shared_k)[vector] = *source;
    }
    __syncthreads();

#pragma unroll
    for (unsigned int dimension_block = 0U;
         dimension_block < kStreamedKktPanel / kSolveSubblock;
         ++dimension_block) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, Bf16,
                     wmma::row_major>
          row_fragment;
      wmma::load_matrix_sync(
          row_fragment,
          shared_k + warp * kSolveSubblock * kStreamedKktPanel +
              dimension_block * kSolveSubblock,
          static_cast<int>(kStreamedKktPanel));
#pragma unroll
      for (unsigned int column_block = 0U; column_block < 4U;
           ++column_block) {
        wmma::fragment<wmma::matrix_b, 16, 16, 16, Bf16,
                       wmma::col_major>
            column_fragment;
        wmma::load_matrix_sync(
            column_fragment,
            shared_k + column_block * kSolveSubblock * kStreamedKktPanel +
                dimension_block * kSolveSubblock,
            static_cast<int>(kStreamedKktPanel));
        wmma::mma_sync(accumulators[column_block], row_fragment,
                       column_fragment, accumulators[column_block]);
      }
    }
    // The next panel overwrites the same 8 KiB bank. The final iteration's
    // barrier also makes that bank available as output scratch.
    __syncthreads();
  }

  auto* const shared_float = reinterpret_cast<float*>(shared_raw);
  float* const warp_scratch =
      shared_float + warp * kSolveSubblock * kSolveSubblock;
  float* const shared_beta = shared_float + kStreamedKktScratchFloats;
  float* const shared_gamma = shared_beta + kChunkSize;
  if (thread < kChunkSize) {
    shared_beta[thread] = matrix_beta[thread];
    shared_gamma[thread] = matrix_gamma[thread];
  }
  __syncthreads();

#pragma unroll
  for (unsigned int column_block = 0U; column_block < 4U;
       ++column_block) {
    wmma::store_matrix_sync(warp_scratch, accumulators[column_block],
                            static_cast<int>(kSolveSubblock),
                            wmma::mem_row_major);
    __syncwarp();
    for (unsigned int index = lane;
         index < kSolveSubblock * kSolveSubblock; index += 32U) {
      const unsigned int row =
          warp * kSolveSubblock + index / kSolveSubblock;
      const unsigned int column =
          column_block * kSolveSubblock + index % kSolveSubblock;
      float value = 0.0F;
      if (row > column) {
        value = shared_beta[row] *
                expf(shared_gamma[row] - shared_gamma[column]) *
                warp_scratch[index];
      }
      matrix_kkt[row * kChunkSize + column] = value;
    }
    __syncwarp();
  }
}

template <unsigned int Row, unsigned int Inner>
__device__ __forceinline__ void accumulate_diagonal_inverse_column(
    const float* const diagonal_l,
    const unsigned int lane,
    const float (&inverse_column)[kSolveSubblock],
    float& value) {
  if constexpr (Inner < Row) {
    if (lane <= Inner) {
      value -= diagonal_l[Row * kChunkSize + Inner] *
               inverse_column[Inner];
    }
    accumulate_diagonal_inverse_column<Row, Inner + 1U>(
        diagonal_l, lane, inverse_column, value);
  }
}

template <unsigned int Row>
__device__ __forceinline__ void solve_diagonal_inverse_column(
    const float* const diagonal_l,
    const unsigned int lane,
    float (&inverse_column)[kSolveSubblock]) {
  float value = lane == Row ? 1.0F : 0.0F;
  accumulate_diagonal_inverse_column<Row, 0U>(
      diagonal_l, lane, inverse_column, value);
  inverse_column[Row] = lane <= Row ? value : 0.0F;
  if constexpr (Row + 1U < kSolveSubblock) {
    solve_diagonal_inverse_column<Row + 1U>(
        diagonal_l, lane, inverse_column);
  }
}

__device__ __forceinline__ void warp_form_transform_block(
    const Bf16* const diagonal_inverse,
    const Bf16* const left0,
    const Bf16* const right0,
    const Bf16* const left1,
    const Bf16* const right1,
    const Bf16* const left2,
    const Bf16* const right2,
    Bf16* const inverse,
    const unsigned int output_base,
    float* const scratch,
    const float* const beta,
    Bf16* const transform,
    const unsigned int row_base,
    const unsigned int column_base) {
  const unsigned int lane = threadIdx.x % 32U;
  WmmaAccumulator inner;
  wmma::fill_fragment(inner, 0.0F);
  warp_accumulate_row_major_product(inner, left0,
                                    static_cast<int>(kChunkSize), right0,
                                    static_cast<int>(kChunkSize));
  if (left1 != nullptr && right1 != nullptr) {
    warp_accumulate_row_major_product(inner, left1,
                                      static_cast<int>(kChunkSize), right1,
                                      static_cast<int>(kChunkSize));
  }
  if (left2 != nullptr && right2 != nullptr) {
    warp_accumulate_row_major_product(inner, left2,
                                      static_cast<int>(kChunkSize), right2,
                                      static_cast<int>(kChunkSize));
  }
  wmma::store_matrix_sync(scratch, inner,
                          static_cast<int>(kSolveSubblock),
                          wmma::mem_row_major);
  __syncwarp();
  for (unsigned int index = lane;
       index < kSolveSubblock * kSolveSubblock; index += 32U) {
    const unsigned int row = index / kSolveSubblock;
    const unsigned int column = index % kSolveSubblock;
    inverse[output_base + row * kChunkSize + column] =
        __float2bfloat16_rn(scratch[index]);
  }
  __syncwarp();

  WmmaAccumulator outer;
  wmma::fill_fragment(outer, 0.0F);
  warp_accumulate_row_major_product(
      outer, diagonal_inverse, static_cast<int>(kChunkSize),
      inverse + output_base, static_cast<int>(kChunkSize));
  wmma::store_matrix_sync(scratch, outer,
                          static_cast<int>(kSolveSubblock),
                          wmma::mem_row_major);
  __syncwarp();
  for (unsigned int index = lane;
       index < kSolveSubblock * kSolveSubblock; index += 32U) {
    const unsigned int row = index / kSolveSubblock;
    const unsigned int column = index % kSolveSubblock;
    const float value = -scratch[index];
    inverse[output_base + row * kChunkSize + column] =
        __float2bfloat16_rn(value);
    const unsigned int output_row = row_base + row;
    const unsigned int output_column = column_base + column;
    transform[output_row + output_column * kChunkSize] =
        __float2bfloat16_rn(value * beta[output_column]);
  }
  __syncwarp();
}

constexpr std::size_t kSplitSolveSharedBytes =
    kChunkSize * kChunkSize *
    (sizeof(float) + 2U * sizeof(Bf16));

static_assert(kSplitSolveSharedBytes == 32U * 1024U);

// The second half mirrors FLA's block-16 merge while preserving the existing
// project's exact FP32-before-final-BF16 transform boundary. Four warps keep
// their diagonal inverse columns in registers. Only BF16 L/inverse operands
// survive into the Tensor-Core block merge; the old full FP32 inverse matrix
// and its 16 KiB shared lifetime disappear.
__global__ __launch_bounds__(kFusedSolveThreads)
void solve_kkt_block16_register_kernel(
    const float* const kkt,
    const float* const beta,
    std::uint16_t* const transform) {
  extern __shared__ unsigned char shared_raw[];
  auto* const shared_l = reinterpret_cast<float*>(shared_raw);
  auto* const shared_l_bf16 = reinterpret_cast<Bf16*>(
      shared_raw + kChunkSize * kChunkSize * sizeof(float));
  auto* const shared_inverse_bf16 =
      shared_l_bf16 + kChunkSize * kChunkSize;
  const unsigned int thread = threadIdx.x;
  const unsigned int warp = thread / 32U;
  const unsigned int lane = thread % 32U;
  const std::size_t matrix = blockIdx.x;
  const float* const matrix_kkt =
      kkt + matrix * kChunkSize * kChunkSize;
  const float* const matrix_beta = beta + matrix * kChunkSize;
  auto* const matrix_transform = reinterpret_cast<Bf16*>(
      transform + matrix * kChunkSize * kChunkSize);

  for (unsigned int index = thread;
       index < kChunkSize * kChunkSize; index += kFusedSolveThreads) {
    shared_l[index] = matrix_kkt[index];
    const unsigned int row = index / kChunkSize;
    const unsigned int column = index % kChunkSize;
    if (row < column) {
      matrix_transform[row + column * kChunkSize] =
          __float2bfloat16_rn(0.0F);
    }
  }
  __syncthreads();

  float inverse_column[kSolveSubblock]{};
  if (lane < kSolveSubblock) {
    const unsigned int diagonal_base =
        warp * kSolveSubblock * kChunkSize + warp * kSolveSubblock;
    solve_diagonal_inverse_column<0U>(
        shared_l + diagonal_base, lane, inverse_column);
#pragma unroll
    for (unsigned int row = 0U; row < kSolveSubblock; ++row) {
      const unsigned int output_row = warp * kSolveSubblock + row;
      const unsigned int output_column = warp * kSolveSubblock + lane;
      const float value = inverse_column[row];
      shared_inverse_bf16[diagonal_base + row * kChunkSize + lane] =
          __float2bfloat16_rn(value);
      matrix_transform[output_row + output_column * kChunkSize] =
          __float2bfloat16_rn(value * matrix_beta[output_column]);
    }
  }
  __syncthreads();

  for (unsigned int index = thread;
       index < kChunkSize * kChunkSize; index += kFusedSolveThreads) {
    shared_l_bf16[index] = __float2bfloat16_rn(shared_l[index]);
  }
  __syncthreads();

  // shared_l is dead after its BF16 publication. Reuse the first 4 KiB for
  // per-warp WMMA scratch and cache beta once in the remaining bank.
  float* const scratch = shared_l;
  float* const shared_beta =
      scratch + kFusedSolveWarps * kSolveSubblock * kSolveSubblock;
  if (thread < kChunkSize) {
    shared_beta[thread] = matrix_beta[thread];
  }
  __syncthreads();
  float* const warp_scratch =
      scratch + warp * kSolveSubblock * kSolveSubblock;

  // First subdiagonal: inv(i,j) = -inv(i,i) L(i,j) inv(j,j).
  if (warp < 3U) {
    const unsigned int row_block = warp + 1U;
    const unsigned int column_block = warp;
    const unsigned int output_base =
        row_block * kSolveSubblock * kChunkSize +
        column_block * kSolveSubblock;
    warp_form_transform_block(
        shared_inverse_bf16 +
            row_block * kSolveSubblock * kChunkSize +
            row_block * kSolveSubblock,
        shared_l_bf16 + output_base,
        shared_inverse_bf16 +
            column_block * kSolveSubblock * kChunkSize +
            column_block * kSolveSubblock,
        nullptr, nullptr, nullptr, nullptr, shared_inverse_bf16,
        output_base, warp_scratch, shared_beta, matrix_transform,
        row_block * kSolveSubblock, column_block * kSolveSubblock);
  }
  __syncthreads();

  if (warp < 2U) {
    const unsigned int row_block = warp + 2U;
    const unsigned int column_block = warp;
    const unsigned int middle_block = warp + 1U;
    const unsigned int output_base =
        row_block * kSolveSubblock * kChunkSize +
        column_block * kSolveSubblock;
    warp_form_transform_block(
        shared_inverse_bf16 +
            row_block * kSolveSubblock * kChunkSize +
            row_block * kSolveSubblock,
        shared_l_bf16 + output_base,
        shared_inverse_bf16 +
            column_block * kSolveSubblock * kChunkSize +
            column_block * kSolveSubblock,
        shared_l_bf16 +
            row_block * kSolveSubblock * kChunkSize +
            middle_block * kSolveSubblock,
        shared_inverse_bf16 +
            middle_block * kSolveSubblock * kChunkSize +
            column_block * kSolveSubblock,
        nullptr, nullptr, shared_inverse_bf16, output_base, warp_scratch,
        shared_beta, matrix_transform, row_block * kSolveSubblock,
        column_block * kSolveSubblock);
  }
  __syncthreads();

  if (warp == 0U) {
    constexpr unsigned int row_block = 3U;
    constexpr unsigned int output_base =
        row_block * kSolveSubblock * kChunkSize;
    warp_form_transform_block(
        shared_inverse_bf16 +
            row_block * kSolveSubblock * kChunkSize +
            row_block * kSolveSubblock,
        shared_l_bf16 + output_base, shared_inverse_bf16,
        shared_l_bf16 + output_base + kSolveSubblock,
        shared_inverse_bf16 + kSolveSubblock * kChunkSize,
        shared_l_bf16 + output_base + 2U * kSolveSubblock,
        shared_inverse_bf16 + 2U * kSolveSubblock * kChunkSize,
        shared_inverse_bf16, output_base, warp_scratch, shared_beta,
        matrix_transform, row_block * kSolveSubblock, 0U);
  }
}

constexpr unsigned int kQkThreads = 128U;
constexpr unsigned int kQkWarps = kQkThreads / 32U;
constexpr unsigned int kWmmaTile = 16U;
constexpr std::size_t kQkSharedBytes =
    2U * kChunkSize * kDimension * sizeof(Bf16) +
    kQkWarps * kWmmaTile * kWmmaTile * sizeof(float);

// FLA output-side QK stage. One CTA owns a complete C64/value-head matrix.
// Q and K are staged once, four warps cover all sixteen 16x16 output tiles,
// and causal gate scaling is applied before the sole BF16 publication. This
// replaces both the external strided GEMM and its global FP32 scale pass.
__global__ __launch_bounds__(kQkThreads)
void qk_scaled_chunk64_kernel(const std::uint16_t* const q,
                              const std::uint16_t* const k,
                              const float* const gamma,
                              std::uint16_t* const qk) {
  extern __shared__ unsigned char shared_raw[];
  auto* const shared_q = reinterpret_cast<Bf16*>(shared_raw);
  auto* const shared_k = shared_q + kChunkSize * kDimension;
  auto* const scratch = reinterpret_cast<float*>(
      shared_k + kChunkSize * kDimension);
  const unsigned int thread = threadIdx.x;
  const unsigned int warp = thread / 32U;
  const unsigned int lane = thread % 32U;
  const std::size_t matrix = blockIdx.x;
  const std::size_t matrix_vector_base =
      matrix * kChunkSize * kDimension;
  const std::size_t matrix_score_base =
      matrix * kChunkSize * kChunkSize;

  for (unsigned int index = thread; index < kChunkSize * kDimension;
       index += kQkThreads) {
    shared_q[index] = reinterpret_cast<const Bf16*>(q)[
        matrix_vector_base + index];
    shared_k[index] = reinterpret_cast<const Bf16*>(k)[
        matrix_vector_base + index];
  }
  __syncthreads();

  for (unsigned int tile = warp; tile < 16U; tile += kQkWarps) {
    const unsigned int query_block = tile / 4U;
    const unsigned int source_block = tile % 4U;
    WmmaAccumulator accumulator;
    wmma::fill_fragment(accumulator, 0.0F);
#pragma unroll
    for (unsigned int key_block = 0U;
         key_block < kDimension / kWmmaTile; ++key_block) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, Bf16,
                     wmma::row_major>
          q_fragment;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, Bf16,
                     wmma::col_major>
          k_fragment;
      wmma::load_matrix_sync(
          q_fragment,
          shared_q + query_block * kWmmaTile * kDimension +
              key_block * kWmmaTile,
          static_cast<int>(kDimension));
      wmma::load_matrix_sync(
          k_fragment,
          shared_k + source_block * kWmmaTile * kDimension +
              key_block * kWmmaTile,
          static_cast<int>(kDimension));
      wmma::mma_sync(accumulator, q_fragment, k_fragment, accumulator);
    }
    float* const tile_scratch =
        scratch + warp * kWmmaTile * kWmmaTile;
    wmma::store_matrix_sync(tile_scratch, accumulator,
                            static_cast<int>(kWmmaTile),
                            wmma::mem_row_major);
    __syncwarp();
    for (unsigned int index = lane; index < kWmmaTile * kWmmaTile;
         index += 32U) {
      const unsigned int query =
          query_block * kWmmaTile + index / kWmmaTile;
      const unsigned int source =
          source_block * kWmmaTile + index % kWmmaTile;
      const float score =
          query >= source
              ? tile_scratch[index] *
                    expf(gamma[matrix * kChunkSize + query] -
                         gamma[matrix * kChunkSize + source])
              : 0.0F;
      qk[matrix_score_base + source * kChunkSize + query] =
          encode_bf16_device(score);
    }
    __syncwarp();
  }
}

// Compact H16 variant. One CTA stages Q/K once for a GQA group, computes the
// raw QK score once, then applies the three value-head-specific gamma vectors
// while publishing the unchanged BF16 QK boundary.
__global__ __launch_bounds__(kQkThreads)
void qk_scaled_group_chunk64_kernel(const std::uint16_t* const compact_q,
                                    const std::uint16_t* const compact_k,
                                    const float* const gamma,
                                    std::uint16_t* const qk) {
  extern __shared__ unsigned char shared_raw[];
  auto* const shared_q = reinterpret_cast<Bf16*>(shared_raw);
  auto* const shared_k = shared_q + kChunkSize * kDimension;
  auto* const scratch = reinterpret_cast<float*>(
      shared_k + kChunkSize * kDimension);
  const unsigned int thread = threadIdx.x;
  const unsigned int warp = thread / 32U;
  const unsigned int lane = thread % 32U;
  const std::size_t compact_matrix = blockIdx.x;
  const std::size_t chunk = compact_matrix / kQkHeadCount;
  const std::size_t qk_head = compact_matrix % kQkHeadCount;
  const std::size_t compact_vector_base =
      compact_matrix * kChunkSize * kDimension;
  const std::size_t first_value_head = qk_head * 3U;

  for (unsigned int index = thread; index < kChunkSize * kDimension;
       index += kQkThreads) {
    shared_q[index] = reinterpret_cast<const Bf16*>(compact_q)[
        compact_vector_base + index];
    shared_k[index] = reinterpret_cast<const Bf16*>(compact_k)[
        compact_vector_base + index];
  }
  __syncthreads();

  for (unsigned int tile = warp; tile < 16U; tile += kQkWarps) {
    const unsigned int query_block = tile / 4U;
    const unsigned int source_block = tile % 4U;
    WmmaAccumulator accumulator;
    wmma::fill_fragment(accumulator, 0.0F);
#pragma unroll
    for (unsigned int key_block = 0U;
         key_block < kDimension / kWmmaTile; ++key_block) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, Bf16,
                     wmma::row_major>
          q_fragment;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, Bf16,
                     wmma::col_major>
          k_fragment;
      wmma::load_matrix_sync(
          q_fragment,
          shared_q + query_block * kWmmaTile * kDimension +
              key_block * kWmmaTile,
          static_cast<int>(kDimension));
      wmma::load_matrix_sync(
          k_fragment,
          shared_k + source_block * kWmmaTile * kDimension +
              key_block * kWmmaTile,
          static_cast<int>(kDimension));
      wmma::mma_sync(accumulator, q_fragment, k_fragment, accumulator);
    }
    float* const tile_scratch =
        scratch + warp * kWmmaTile * kWmmaTile;
    wmma::store_matrix_sync(tile_scratch, accumulator,
                            static_cast<int>(kWmmaTile),
                            wmma::mem_row_major);
    __syncwarp();
    for (unsigned int index = lane; index < kWmmaTile * kWmmaTile;
         index += 32U) {
      const unsigned int query =
          query_block * kWmmaTile + index / kWmmaTile;
      const unsigned int source =
          source_block * kWmmaTile + index % kWmmaTile;
#pragma unroll
      for (unsigned int replica = 0U; replica < 3U; ++replica) {
        const std::size_t value_head = first_value_head + replica;
        const std::size_t matrix = chunk * kValueHeadCount + value_head;
        const float score =
            query >= source
                ? tile_scratch[index] *
                      expf(gamma[matrix * kChunkSize + query] -
                           gamma[matrix * kChunkSize + source])
                : 0.0F;
        qk[matrix * kChunkSize * kChunkSize +
           source * kChunkSize + query] = encode_bf16_device(score);
      }
    }
    __syncwarp();
  }
}

constexpr unsigned int kWuThreads = 256U;
constexpr unsigned int kWuWarps = kWuThreads / 32U;
constexpr std::size_t kWuSharedBytes =
    (kChunkSize * kChunkSize +
     2U * kChunkSize * kDimension) * sizeof(Bf16) +
    kWuWarps * kWmmaTile * kWmmaTile * sizeof(float);

// FLA WY recomputation stage. The transform, exp(g)K and V are each loaded
// exactly once per C64/value-head CTA. Eight warps own independent 16-column
// slabs and produce W and U with BF16 Tensor Cores, preserving the screened
// BF16 boundary consumed by the persistent-state stage.
__global__ __launch_bounds__(kWuThreads)
void recompute_w_u_chunk64_kernel(
    const std::uint16_t* const transform,
    const std::uint16_t* const k_g,
    const std::uint16_t* const v,
    std::uint16_t* const w,
    std::uint16_t* const u) {
  extern __shared__ unsigned char shared_raw[];
  auto* const shared_transform = reinterpret_cast<Bf16*>(shared_raw);
  auto* const shared_k = shared_transform + kChunkSize * kChunkSize;
  auto* const shared_v = shared_k + kChunkSize * kDimension;
  auto* const scratch = reinterpret_cast<float*>(
      shared_v + kChunkSize * kDimension);
  const unsigned int thread = threadIdx.x;
  const unsigned int warp = thread / 32U;
  const unsigned int lane = thread % 32U;
  const std::size_t matrix = blockIdx.x;
  const std::size_t matrix_transform_base =
      matrix * kChunkSize * kChunkSize;
  const std::size_t matrix_vector_base =
      matrix * kChunkSize * kDimension;

  for (unsigned int index = thread; index < kChunkSize * kChunkSize;
       index += kWuThreads) {
    shared_transform[index] = reinterpret_cast<const Bf16*>(transform)[
        matrix_transform_base + index];
  }
  for (unsigned int index = thread; index < kChunkSize * kDimension;
       index += kWuThreads) {
    shared_k[index] =
        reinterpret_cast<const Bf16*>(k_g)[matrix_vector_base + index];
    shared_v[index] =
        reinterpret_cast<const Bf16*>(v)[matrix_vector_base + index];
  }
  __syncthreads();

  float* const tile_scratch =
      scratch + warp * kWmmaTile * kWmmaTile;
#pragma unroll
  for (unsigned int token_block = 0U;
       token_block < kChunkSize / kWmmaTile; ++token_block) {
    WmmaAccumulator w_accumulator;
    WmmaAccumulator u_accumulator;
    wmma::fill_fragment(w_accumulator, 0.0F);
    wmma::fill_fragment(u_accumulator, 0.0F);
#pragma unroll
    for (unsigned int source_block = 0U;
         source_block < kChunkSize / kWmmaTile; ++source_block) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, Bf16,
                     wmma::col_major>
          transform_fragment;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, Bf16,
                     wmma::row_major>
          k_fragment;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, Bf16,
                     wmma::row_major>
          v_fragment;
      wmma::load_matrix_sync(
          transform_fragment,
          shared_transform + token_block * kWmmaTile +
              source_block * kWmmaTile * kChunkSize,
          static_cast<int>(kChunkSize));
      wmma::load_matrix_sync(
          k_fragment,
          shared_k + source_block * kWmmaTile * kDimension +
              warp * kWmmaTile,
          static_cast<int>(kDimension));
      wmma::load_matrix_sync(
          v_fragment,
          shared_v + source_block * kWmmaTile * kDimension +
              warp * kWmmaTile,
          static_cast<int>(kDimension));
      wmma::mma_sync(w_accumulator, transform_fragment, k_fragment,
                     w_accumulator);
      wmma::mma_sync(u_accumulator, transform_fragment, v_fragment,
                     u_accumulator);
    }

    wmma::store_matrix_sync(tile_scratch, w_accumulator,
                            static_cast<int>(kWmmaTile),
                            wmma::mem_row_major);
    __syncwarp();
    for (unsigned int index = lane; index < kWmmaTile * kWmmaTile;
         index += 32U) {
      const unsigned int row = index / kWmmaTile;
      const unsigned int column = index % kWmmaTile;
      w[matrix_vector_base +
        (token_block * kWmmaTile + row) * kDimension +
        warp * kWmmaTile + column] = encode_bf16_device(tile_scratch[index]);
    }
    __syncwarp();
    wmma::store_matrix_sync(tile_scratch, u_accumulator,
                            static_cast<int>(kWmmaTile),
                            wmma::mem_row_major);
    __syncwarp();
    for (unsigned int index = lane; index < kWmmaTile * kWmmaTile;
         index += 32U) {
      const unsigned int row = index / kWmmaTile;
      const unsigned int column = index % kWmmaTile;
      u[matrix_vector_base +
        (token_block * kWmmaTile + row) * kDimension +
        warp * kWmmaTile + column] = encode_bf16_device(tile_scratch[index]);
    }
    __syncwarp();
  }
}

// Frozen stage-B baseline. It keeps the FP32 recurrence in accumulator
// fragments, but each chunk still materializes the entire 64x128 tile as
// FP32, rounds it into a second BF16 shared tile, and reloads that tile for W.
__global__ __launch_bounds__(kFusedSolveThreads)
void persistent_state_chunk64_baseline_kernel(
    const std::uint16_t* const w,
    const std::uint16_t* const u,
    const std::uint16_t* const k_decay,
    const float* const gamma,
    const std::uint16_t* const state_input,
    std::uint16_t* const state_output,
    const unsigned int chunk_count,
    std::uint16_t* const v_new,
    std::uint16_t* const boundary_state) {
  extern __shared__ unsigned char shared_raw[];
  auto* const shared_state_fp32 = reinterpret_cast<float*>(shared_raw);
  auto* const shared_state_bf16 = reinterpret_cast<Bf16*>(
      shared_raw + 64U * kDimension * sizeof(float));

  const unsigned int thread = threadIdx.x;
  const unsigned int warp = thread / 32U;
  const unsigned int lane = thread % 32U;
  const unsigned int value_head = blockIdx.x / 2U;
  const unsigned int value_half = blockIdx.x % 2U;
  const unsigned int local_value_base = warp * kSolveSubblock;
  const unsigned int value_base =
      value_half * 64U + local_value_base;
  const std::size_t head_state_base =
      static_cast<std::size_t>(value_head) * kDimension * kDimension;

  for (unsigned int index = thread; index < 64U * kDimension;
       index += kFusedSolveThreads) {
    const unsigned int local_value = index / kDimension;
    const unsigned int key_dimension = index % kDimension;
    const std::size_t source =
        head_state_base +
        static_cast<std::size_t>(value_half * 64U + local_value) *
            kDimension +
        key_dimension;
    shared_state_fp32[index] =
        decode_bf16_device(state_input[source]);
  }
  __syncthreads();

  WmmaAccumulator state_fragments[kDimension / kSolveSubblock];
#pragma unroll
  for (unsigned int key_block = 0U;
       key_block < kDimension / kSolveSubblock; ++key_block) {
    wmma::load_matrix_sync(
        state_fragments[key_block],
        shared_state_fp32 + local_value_base * kDimension +
            key_block * kSolveSubblock,
        static_cast<int>(kDimension), wmma::mem_row_major);
  }

  for (unsigned int chunk_index = 0U; chunk_index < chunk_count;
       ++chunk_index) {
#pragma unroll
    for (unsigned int key_block = 0U;
         key_block < kDimension / kSolveSubblock; ++key_block) {
      wmma::store_matrix_sync(
          shared_state_fp32 + local_value_base * kDimension +
              key_block * kSolveSubblock,
          state_fragments[key_block], static_cast<int>(kDimension),
          wmma::mem_row_major);
    }
    __syncthreads();

    const std::size_t matrix =
        static_cast<std::size_t>(chunk_index) * kValueHeadCount +
        value_head;
    for (unsigned int index = thread; index < 64U * kDimension;
         index += kFusedSolveThreads) {
      const unsigned int local_value = index / kDimension;
      const unsigned int key_dimension = index % kDimension;
      const Bf16 rounded =
          __float2bfloat16_rn(shared_state_fp32[index]);
      shared_state_bf16[index] = rounded;
      boundary_state[
          matrix * kDimension * kDimension +
          static_cast<std::size_t>(value_half * 64U + local_value) *
              kDimension +
          key_dimension] = __bfloat16_as_ushort(rounded);
    }
    __syncthreads();

    const auto* const matrix_w = reinterpret_cast<const Bf16*>(
        w + matrix * kChunkSize * kDimension);
    const auto* const matrix_u = reinterpret_cast<const Bf16*>(
        u + matrix * kChunkSize * kDimension);
    auto* const matrix_v_new = reinterpret_cast<Bf16*>(
        v_new + matrix * kChunkSize * kDimension);

#pragma unroll
    for (unsigned int token_block = 0U;
         token_block < kChunkSize / kSolveSubblock; ++token_block) {
      WmmaAccumulator product;
      wmma::fill_fragment(product, 0.0F);
#pragma unroll
      for (unsigned int key_block = 0U;
           key_block < kDimension / kSolveSubblock; ++key_block) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, Bf16,
                       wmma::row_major>
            state_fragment;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, Bf16,
                       wmma::col_major>
            w_fragment;
        wmma::load_matrix_sync(
            state_fragment,
            shared_state_bf16 + local_value_base * kDimension +
                key_block * kSolveSubblock,
            static_cast<int>(kDimension));
        wmma::load_matrix_sync(
            w_fragment,
            matrix_w + token_block * kSolveSubblock * kDimension +
                key_block * kSolveSubblock,
            static_cast<int>(kDimension));
        wmma::mma_sync(product, state_fragment, w_fragment, product);
      }
      float* const product_scratch =
          shared_state_fp32 + warp * kSolveSubblock * kSolveSubblock;
      wmma::store_matrix_sync(product_scratch, product,
                              static_cast<int>(kSolveSubblock),
                              wmma::mem_row_major);
      __syncwarp();
      for (unsigned int index = lane;
           index < kSolveSubblock * kSolveSubblock; index += 32U) {
        const unsigned int local_value = index / kSolveSubblock;
        const unsigned int token_in_block = index % kSolveSubblock;
        const unsigned int value_dimension = value_base + local_value;
        const unsigned int token_in_chunk =
            token_block * kSolveSubblock + token_in_block;
        const std::size_t element =
            static_cast<std::size_t>(token_in_chunk) * kDimension +
            value_dimension;
        const float corrected = __bfloat162float(matrix_u[element]) -
                                product_scratch[index];
        matrix_v_new[element] = __float2bfloat16_rn(corrected);
      }
      __syncwarp();
    }
    __syncthreads();

    const float decay =
        expf(gamma[matrix * kChunkSize + kChunkSize - 1U]);
#pragma unroll
    for (unsigned int key_block = 0U;
         key_block < kDimension / kSolveSubblock; ++key_block) {
#pragma unroll
      for (unsigned int element = 0U;
           element < state_fragments[key_block].num_elements; ++element) {
        state_fragments[key_block].x[element] *= decay;
      }
    }

    const auto* const matrix_k_decay = reinterpret_cast<const Bf16*>(
        k_decay + matrix * kChunkSize * kDimension);
#pragma unroll
    for (unsigned int key_block = 0U;
         key_block < kDimension / kSolveSubblock; ++key_block) {
#pragma unroll
      for (unsigned int token_block = 0U;
           token_block < kChunkSize / kSolveSubblock; ++token_block) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, Bf16,
                       wmma::col_major>
            v_fragment;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, Bf16,
                       wmma::row_major>
            k_fragment;
        wmma::load_matrix_sync(
            v_fragment,
            matrix_v_new + token_block * kSolveSubblock * kDimension +
                value_base,
            static_cast<int>(kDimension));
        wmma::load_matrix_sync(
            k_fragment,
            matrix_k_decay +
                token_block * kSolveSubblock * kDimension +
                key_block * kSolveSubblock,
            static_cast<int>(kDimension));
        wmma::mma_sync(state_fragments[key_block], v_fragment, k_fragment,
                       state_fragments[key_block]);
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (unsigned int key_block = 0U;
       key_block < kDimension / kSolveSubblock; ++key_block) {
    wmma::store_matrix_sync(
        shared_state_fp32 + local_value_base * kDimension +
            key_block * kSolveSubblock,
        state_fragments[key_block], static_cast<int>(kDimension),
        wmma::mem_row_major);
  }
  __syncthreads();
  for (unsigned int index = thread; index < 64U * kDimension;
       index += kFusedSolveThreads) {
    const unsigned int local_value = index / kDimension;
    const unsigned int key_dimension = index % kDimension;
    const std::size_t destination =
        head_state_base +
        static_cast<std::size_t>(value_half * 64U + local_value) *
            kDimension +
        key_dimension;
    state_output[destination] =
        encode_bf16_device(shared_state_fp32[index]);
  }
}

constexpr std::size_t kResidentStateScratchElements =
    kFusedSolveWarps * kSolveSubblock * kSolveSubblock;
constexpr std::size_t kResidentStateScratchBytes =
    kResidentStateScratchElements * sizeof(float);
constexpr std::size_t kResidentVNewElements = 64U * kChunkSize;
constexpr std::size_t kResidentVNewBytes =
    kResidentVNewElements * sizeof(Bf16);
constexpr std::size_t kResidentMatrixElements = kChunkSize * kDimension;
constexpr std::size_t kResidentMatrixBytes =
    kResidentMatrixElements * sizeof(Bf16);
constexpr std::size_t kResidentStateSharedBytes =
    kResidentStateScratchBytes + kResidentVNewBytes +
    kResidentMatrixBytes;
constexpr std::size_t kResidentDecayScaleBytes =
    kChunkSize * sizeof(float);
constexpr std::size_t kResidentStatePacklessSharedBytes =
    kResidentStateSharedBytes + kResidentDecayScaleBytes;
static_assert(kResidentStateSharedBytes == 28U * 1024U);
static_assert(kResidentStatePacklessSharedBytes == 28U * 1024U + 256U);

__device__ __forceinline__ void resident_state_cp_async_16(
    void* const shared_destination, const void* const global_source) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const unsigned int shared_address =
      static_cast<unsigned int>(__cvta_generic_to_shared(
          shared_destination));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :
               : "r"(shared_address), "l"(global_source));
#endif
}

__device__ __forceinline__ void resident_state_cp_async_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;" ::: "memory");
#endif
}

__device__ __forceinline__ void resident_state_cp_async_wait_all() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group 0;" ::: "memory");
#endif
}

// Open-book stage-B candidate, specialized to the authenticated GDN shape.
// The recurrence H remains in FP32 accumulator fragments for the whole
// request. At each chunk, its BF16 matrix-A operand is formed directly from
// those fragments, rather than through the baseline's 32-KiB FP32 and
// 16-KiB BF16 shared round trip. One 16-KiB shared slot is reused for W then
// K, so every global tile is loaded once per CTA instead of once per warp.
// The CTA-owned 64-row half of Vnew is retained in 8 KiB of shared memory for
// the state update while the public BF16 Vnew boundary is still written.
template <bool Packless>
__global__ __launch_bounds__(kFusedSolveThreads)
void persistent_state_chunk64_resident_kernel(
    const std::uint16_t* const w,
    const std::uint16_t* const u,
    const std::uint16_t* const k_decay,
    const float* const gamma,
    const std::uint16_t* const state_input,
    std::uint16_t* const state_output,
    const unsigned int chunk_count,
    std::uint16_t* const v_new,
    std::uint16_t* const boundary_state) {
  extern __shared__ unsigned char shared_raw[];
  auto* const shared_scratch = reinterpret_cast<float*>(shared_raw);
  auto* const shared_v_new = reinterpret_cast<Bf16*>(
      shared_raw + kResidentStateScratchBytes);
  auto* const shared_matrix = reinterpret_cast<Bf16*>(
      shared_raw + kResidentStateScratchBytes + kResidentVNewBytes);
  auto* const shared_decay_scale = reinterpret_cast<float*>(
      shared_raw + kResidentStateSharedBytes);

  const unsigned int thread = threadIdx.x;
  const unsigned int warp = thread / 32U;
  const unsigned int lane = thread % 32U;
  const unsigned int value_head = blockIdx.x / 2U;
  const unsigned int value_half = blockIdx.x % 2U;
  const unsigned int local_value_base = warp * kSolveSubblock;
  const unsigned int value_base = value_half * 64U + local_value_base;
  const std::size_t head_state_base =
      static_cast<std::size_t>(value_head) * kDimension * kDimension;
  float* const tile_scratch =
      shared_scratch + warp * kSolveSubblock * kSolveSubblock;

  WmmaAccumulator state_fragments[kDimension / kSolveSubblock];
#pragma unroll
  for (unsigned int key_block = 0U;
       key_block < kDimension / kSolveSubblock; ++key_block) {
    for (unsigned int index = lane;
         index < kSolveSubblock * kSolveSubblock; index += 32U) {
      const unsigned int local_value = index / kSolveSubblock;
      const unsigned int key_in_block = index % kSolveSubblock;
      const std::size_t source =
          head_state_base +
          static_cast<std::size_t>(value_base + local_value) * kDimension +
          key_block * kSolveSubblock + key_in_block;
      tile_scratch[index] = decode_bf16_device(state_input[source]);
    }
    __syncwarp();
    wmma::load_matrix_sync(state_fragments[key_block], tile_scratch,
                           static_cast<int>(kSolveSubblock),
                           wmma::mem_row_major);
    __syncwarp();
  }

  for (unsigned int chunk_index = 0U; chunk_index < chunk_count;
       ++chunk_index) {
    const std::size_t matrix =
        static_cast<std::size_t>(chunk_index) * kValueHeadCount +
        value_head;
    const auto* const matrix_w = reinterpret_cast<const Bf16*>(
        w + matrix * kChunkSize * kDimension);
    const auto* const matrix_u = reinterpret_cast<const Bf16*>(
        u + matrix * kChunkSize * kDimension);
    const std::size_t k_matrix =
        Packless ? static_cast<std::size_t>(chunk_index) * kQkHeadCount +
                       value_head / 3U
                 : matrix;
    const auto* const matrix_k_source = reinterpret_cast<const Bf16*>(
        k_decay + k_matrix * kChunkSize * kDimension);
    auto* const matrix_v_new = reinterpret_cast<Bf16*>(
        v_new + matrix * kChunkSize * kDimension);

    // Hide the once-per-CTA W transfer behind the mandatory H publication.
    for (unsigned int vector = thread;
         vector < kResidentMatrixBytes / 16U;
         vector += kFusedSolveThreads) {
      resident_state_cp_async_16(
          reinterpret_cast<unsigned char*>(shared_matrix) + vector * 16U,
          reinterpret_cast<const unsigned char*>(matrix_w) + vector * 16U);
    }
    resident_state_cp_async_commit();

#pragma unroll
    for (unsigned int key_block = 0U;
         key_block < kDimension / kSolveSubblock; ++key_block) {
      wmma::store_matrix_sync(tile_scratch, state_fragments[key_block],
                              static_cast<int>(kSolveSubblock),
                              wmma::mem_row_major);
      __syncwarp();
      for (unsigned int index = lane;
           index < kSolveSubblock * kSolveSubblock; index += 32U) {
        const unsigned int local_value = index / kSolveSubblock;
        const unsigned int key_in_block = index % kSolveSubblock;
        boundary_state[
            matrix * kDimension * kDimension +
            static_cast<std::size_t>(value_base + local_value) * kDimension +
            key_block * kSolveSubblock + key_in_block] =
            encode_bf16_device(tile_scratch[index]);
      }
      __syncwarp();
    }

    resident_state_cp_async_wait_all();
    __syncthreads();

    // Four products remain live long enough to free the shared W/K slot.
    WmmaAccumulator products[kChunkSize / kSolveSubblock];
#pragma unroll
    for (unsigned int token_block = 0U;
         token_block < kChunkSize / kSolveSubblock; ++token_block) {
      wmma::fill_fragment(products[token_block], 0.0F);
#pragma unroll
      for (unsigned int key_block = 0U;
           key_block < kDimension / kSolveSubblock; ++key_block) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, Bf16,
                       wmma::row_major>
            state_fragment;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, Bf16,
                       wmma::col_major>
            w_fragment;
#pragma unroll
        for (unsigned int element = 0U;
             element < state_fragment.num_elements; ++element) {
          state_fragment.x[element] = __float2bfloat16_rn(
              state_fragments[key_block].x[element]);
        }
        wmma::load_matrix_sync(
            w_fragment,
            shared_matrix +
                token_block * kSolveSubblock * kDimension +
                key_block * kSolveSubblock,
            static_cast<int>(kDimension));
        wmma::mma_sync(products[token_block], state_fragment, w_fragment,
                       products[token_block]);
      }
    }
    __syncthreads();

    // Reuse the same slot for K. The packed baseline can copy its preformed
    // BF16 boundary asynchronously. Production instead forms the identical
    // end-decayed BF16 K from compact H16 K and gamma at the consumption
    // point, eliminating the global materialization and its 3x GQA copies.
    if constexpr (Packless) {
      if (thread < kChunkSize) {
        shared_decay_scale[thread] = expf(
            gamma[matrix * kChunkSize + kChunkSize - 1U] -
            gamma[matrix * kChunkSize + thread]);
      }
      __syncthreads();
      for (unsigned int index = thread;
           index < kResidentMatrixElements;
           index += kFusedSolveThreads) {
        const unsigned int token_in_chunk = index / kDimension;
        shared_matrix[index] = __float2bfloat16_rn(
            shared_decay_scale[token_in_chunk] *
            __bfloat162float(matrix_k_source[index]));
      }
      __syncthreads();
    } else {
      for (unsigned int vector = thread;
           vector < kResidentMatrixBytes / 16U;
           vector += kFusedSolveThreads) {
        resident_state_cp_async_16(
            reinterpret_cast<unsigned char*>(shared_matrix) + vector * 16U,
            reinterpret_cast<const unsigned char*>(matrix_k_source) +
                vector * 16U);
      }
      resident_state_cp_async_commit();
    }

#pragma unroll
    for (unsigned int token_block = 0U;
         token_block < kChunkSize / kSolveSubblock; ++token_block) {
      wmma::store_matrix_sync(tile_scratch, products[token_block],
                              static_cast<int>(kSolveSubblock),
                              wmma::mem_row_major);
      __syncwarp();
      for (unsigned int index = lane;
           index < kSolveSubblock * kSolveSubblock; index += 32U) {
        const unsigned int local_value = index / kSolveSubblock;
        const unsigned int token_in_block = index % kSolveSubblock;
        const unsigned int token_in_chunk =
            token_block * kSolveSubblock + token_in_block;
        const unsigned int local_half_value =
            local_value_base + local_value;
        const unsigned int value_dimension =
            value_half * 64U + local_half_value;
        const std::size_t element =
            static_cast<std::size_t>(token_in_chunk) * kDimension +
            value_dimension;
        const Bf16 corrected = __float2bfloat16_rn(
            __bfloat162float(matrix_u[element]) - tile_scratch[index]);
        matrix_v_new[element] = corrected;
        shared_v_new[token_in_chunk * 64U + local_half_value] = corrected;
      }
      __syncwarp();
    }

    const float decay =
        expf(gamma[matrix * kChunkSize + kChunkSize - 1U]);
#pragma unroll
    for (unsigned int key_block = 0U;
         key_block < kDimension / kSolveSubblock; ++key_block) {
#pragma unroll
      for (unsigned int element = 0U;
           element < state_fragments[key_block].num_elements; ++element) {
        state_fragments[key_block].x[element] *= decay;
      }
    }

    if constexpr (!Packless) {
      resident_state_cp_async_wait_all();
    }
    __syncthreads();

#pragma unroll
    for (unsigned int key_block = 0U;
         key_block < kDimension / kSolveSubblock; ++key_block) {
#pragma unroll
      for (unsigned int token_block = 0U;
           token_block < kChunkSize / kSolveSubblock; ++token_block) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, Bf16,
                       wmma::col_major>
            v_fragment;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, Bf16,
                       wmma::row_major>
            k_fragment;
        wmma::load_matrix_sync(
            v_fragment,
            shared_v_new +
                token_block * kSolveSubblock * 64U + local_value_base,
            64);
        wmma::load_matrix_sync(
            k_fragment,
            shared_matrix +
                token_block * kSolveSubblock * kDimension +
                key_block * kSolveSubblock,
            static_cast<int>(kDimension));
        wmma::mma_sync(state_fragments[key_block], v_fragment, k_fragment,
                       state_fragments[key_block]);
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (unsigned int key_block = 0U;
       key_block < kDimension / kSolveSubblock; ++key_block) {
    wmma::store_matrix_sync(tile_scratch, state_fragments[key_block],
                            static_cast<int>(kSolveSubblock),
                            wmma::mem_row_major);
    __syncwarp();
    for (unsigned int index = lane;
         index < kSolveSubblock * kSolveSubblock; index += 32U) {
      const unsigned int local_value = index / kSolveSubblock;
      const unsigned int key_in_block = index % kSolveSubblock;
      const std::size_t destination =
          head_state_base +
          static_cast<std::size_t>(value_base + local_value) * kDimension +
          key_block * kSolveSubblock + key_in_block;
      state_output[destination] =
          encode_bf16_device(tile_scratch[index]);
    }
    __syncwarp();
  }
}

// Frozen-vLLM-faithful state recurrence candidate for SM87.  The resident
// state has the same logical [BV64, BK64] shape as the public H boundary, but
// its physical register ownership follows the update GEMM accumulator:
// every warp owns K16 x V64 (eight m16n8 fragments) in each K64 bank.  W*H
// temporarily changes ownership to T16 x V64 through one shared panel; the
// K*V update then lands directly back in the resident state fragments.
//
// The six 8-KiB panels and 260-byte gate tail reproduce the authenticated
// vLLM specialization's 49,412-byte lifetime.  W/U/raw-K/gamma are issued
// with cp.async before H publication, while the sixth panel is reused for
// the state operand and rounded Vnew.  The production selector remains the
// resident kernel above; this kernel is reachable only through the explicit
// diagnostic selector below.
constexpr unsigned int kFaithfulPanelRows = 64U;
constexpr unsigned int kFaithfulPanelColumns = 64U;
constexpr unsigned int kFaithfulPanelChunks = 8U;
constexpr unsigned int kFaithfulKeyBanks = 2U;
constexpr unsigned int kFaithfulValuePanels = 8U;
constexpr std::size_t kFaithfulPanelElements =
    kFaithfulPanelRows * kFaithfulPanelColumns;
constexpr std::size_t kFaithfulPanelBytes =
    kFaithfulPanelElements * sizeof(std::uint16_t);
constexpr std::size_t kFaithfulW0Offset = 0U;
constexpr std::size_t kFaithfulW1Offset =
    kFaithfulW0Offset + kFaithfulPanelBytes;
constexpr std::size_t kFaithfulUOffset =
    kFaithfulW1Offset + kFaithfulPanelBytes;
constexpr std::size_t kFaithfulK0Offset =
    kFaithfulUOffset + kFaithfulPanelBytes;
constexpr std::size_t kFaithfulK1Offset =
    kFaithfulK0Offset + kFaithfulPanelBytes;
constexpr std::size_t kFaithfulTransientOffset =
    kFaithfulK1Offset + kFaithfulPanelBytes;
constexpr std::size_t kFaithfulGammaOffset =
    kFaithfulTransientOffset + kFaithfulPanelBytes;
constexpr std::size_t kFaithfulGammaElements = kChunkSize + 1U;
constexpr std::size_t kFaithfulStateSharedBytes =
    kFaithfulGammaOffset + kFaithfulGammaElements * sizeof(float);
static_assert(kFaithfulGammaOffset == 48U * 1024U);
static_assert(kFaithfulStateSharedBytes == 49412U);

struct FaithfulM16K16 final {
  std::uint32_t x0;
  std::uint32_t x1;
  std::uint32_t x2;
  std::uint32_t x3;
};

struct FaithfulK16N8 final {
  std::uint32_t x0;
  std::uint32_t x1;
};

struct FaithfulM16N8Accumulator final {
  float x0;
  float x1;
  float x2;
  float x3;
};

[[nodiscard]] __device__ __forceinline__ unsigned int
faithful_swizzled_chunk(const unsigned int row,
                        const unsigned int logical_chunk) {
  return logical_chunk ^ (row & 7U);
}

[[nodiscard]] __device__ __forceinline__ std::uint32_t
faithful_pack_pair(const float low, const float high) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  std::uint32_t packed = 0U;
  asm("cvt.rn.bf16x2.f32 %0, %2, %1;"
      : "=r"(packed)
      : "f"(low), "f"(high));
  return packed;
#else
  return static_cast<std::uint32_t>(encode_bf16_device(low)) |
         (static_cast<std::uint32_t>(encode_bf16_device(high)) << 16U);
#endif
}

__device__ __forceinline__ void faithful_zero(
    FaithfulM16N8Accumulator& accumulator) {
  accumulator.x0 = 0.0F;
  accumulator.x1 = 0.0F;
  accumulator.x2 = 0.0F;
  accumulator.x3 = 0.0F;
}

__device__ __forceinline__ void faithful_mma(
    FaithfulM16N8Accumulator& accumulator,
    const FaithfulM16K16& a,
    const FaithfulK16N8& b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(accumulator.x0), "+f"(accumulator.x1),
        "+f"(accumulator.x2), "+f"(accumulator.x3)
      : "r"(a.x0), "r"(a.x1), "r"(a.x2), "r"(a.x3),
        "r"(b.x0), "r"(b.x1));
#else
  (void)accumulator;
  (void)a;
  (void)b;
#endif
}

// Row-major [M64,K64] matrix-A loader.  Four warps select independent M16
// rows; the XOR swizzle is identical to the vLLM shared layout.
__device__ __forceinline__ void faithful_load_a(
    FaithfulM16K16& fragment,
    const std::uint16_t* const shared_matrix,
    const unsigned int m_panel,
    const unsigned int k16,
    const unsigned int lane) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
  const unsigned int quadrant = lane / 8U;
  const unsigned int row =
      m_panel * 16U + (lane % 8U) + (quadrant & 1U) * 8U;
  const unsigned int logical_chunk =
      k16 * 2U + (quadrant >> 1U);
  const unsigned int physical_chunk =
      faithful_swizzled_chunk(row, logical_chunk);
  const auto* const source =
      shared_matrix + row * kFaithfulPanelColumns +
      physical_chunk * 8U;
  const unsigned int shared_address = static_cast<unsigned int>(
      __cvta_generic_to_shared(source));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(fragment.x0), "=r"(fragment.x1),
        "=r"(fragment.x2), "=r"(fragment.x3)
      : "r"(shared_address)
      : "memory");
#else
  (void)fragment;
  (void)shared_matrix;
  (void)m_panel;
  (void)k16;
  (void)lane;
#endif
}

// Load two adjacent K16 matrix-B fragments from the row-major [N64,K64]
// backing of a column-major logical [K64,N64] operand. Each canonical 8x8
// tile is already the transpose of true [K,N], so a plain x4 load is the
// register-equivalent pair of true-[K,N] x2.trans loads.
__device__ __forceinline__ void faithful_load_b_pair_transposed(
    FaithfulK16N8& first,
    FaithfulK16N8& second,
    const std::uint16_t* const shared_matrix,
    const unsigned int n_panel,
    const unsigned int k32,
    const unsigned int lane) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
  const unsigned int row = n_panel * 8U + (lane & 7U);
  const unsigned int logical_chunk = k32 * 4U + (lane >> 3U);
  const unsigned int physical_chunk =
      faithful_swizzled_chunk(row, logical_chunk);
  const auto* const source =
      shared_matrix + row * kFaithfulPanelColumns +
      physical_chunk * 8U;
  const unsigned int shared_address = static_cast<unsigned int>(
      __cvta_generic_to_shared(source));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(first.x0), "=r"(first.x1), "=r"(second.x0),
        "=r"(second.x1)
      : "r"(shared_address)
      : "memory");
#else
  (void)first;
  (void)second;
  (void)shared_matrix;
  (void)n_panel;
  (void)k32;
  (void)lane;
#endif
}

__device__ __forceinline__ void faithful_stage_panel(
    const std::uint16_t* const global_matrix,
    const unsigned int global_leading,
    const unsigned int global_column,
    std::uint16_t* const shared_matrix,
    const unsigned int thread) {
  constexpr unsigned int kVectorCount =
      kFaithfulPanelBytes / sizeof(uint4);
  static_assert(kVectorCount == 512U);
  for (unsigned int vector = thread; vector < kVectorCount;
       vector += kFusedSolveThreads) {
    const unsigned int row = vector / kFaithfulPanelChunks;
    const unsigned int logical_chunk =
        vector % kFaithfulPanelChunks;
    const unsigned int physical_chunk =
        faithful_swizzled_chunk(row, logical_chunk);
    resident_state_cp_async_16(
        shared_matrix + row * kFaithfulPanelColumns +
            physical_chunk * 8U,
        global_matrix + row * global_leading + global_column +
            logical_chunk * 8U);
  }
  resident_state_cp_async_commit();
}

__device__ __forceinline__ void faithful_stage_chunk(
    const std::uint16_t* const matrix_w,
    const std::uint16_t* const matrix_u,
    const std::uint16_t* const matrix_k,
    const float* const matrix_gamma,
    const unsigned int value_half,
    std::uint16_t* const shared_w0,
    std::uint16_t* const shared_w1,
    std::uint16_t* const shared_u,
    std::uint16_t* const shared_k0,
    std::uint16_t* const shared_k1,
    float* const shared_gamma,
    const unsigned int thread) {
  faithful_stage_panel(matrix_w, kDimension, 0U, shared_w0, thread);
  faithful_stage_panel(matrix_w, kDimension, 64U, shared_w1, thread);
  faithful_stage_panel(matrix_u, kDimension, value_half * 64U,
                       shared_u, thread);
  faithful_stage_panel(matrix_k, kDimension, 0U, shared_k0, thread);
  faithful_stage_panel(matrix_k, kDimension, 64U, shared_k1, thread);
  if (thread < kChunkSize * sizeof(float) / 16U) {
    resident_state_cp_async_16(shared_gamma + thread * 4U,
                               matrix_gamma + thread * 4U);
  }
  resident_state_cp_async_commit();
}

__device__ __forceinline__ void faithful_load_initial_state(
    FaithfulM16N8Accumulator
        (&state)[kFaithfulKeyBanks][kFaithfulValuePanels],
    const std::uint16_t* const state_input,
    const std::size_t head_state_base,
    const unsigned int value_half,
    const unsigned int warp,
    const unsigned int lane) {
  const unsigned int group = lane >> 2U;
  const unsigned int pair = lane & 3U;
  const unsigned int key0 = warp * 16U + group;
  const unsigned int key1 = key0 + 8U;
#pragma unroll
  for (unsigned int bank = 0U; bank < kFaithfulKeyBanks; ++bank) {
#pragma unroll
    for (unsigned int panel = 0U; panel < kFaithfulValuePanels;
         ++panel) {
      const unsigned int value0 =
          value_half * 64U + panel * 8U + pair * 2U;
      const unsigned int value1 = value0 + 1U;
      const unsigned int bank_key0 = bank * 64U + key0;
      const unsigned int bank_key1 = bank * 64U + key1;
      state[bank][panel].x0 = decode_bf16_device(
          state_input[head_state_base +
                      static_cast<std::size_t>(value0) * kDimension +
                      bank_key0]);
      state[bank][panel].x1 = decode_bf16_device(
          state_input[head_state_base +
                      static_cast<std::size_t>(value1) * kDimension +
                      bank_key0]);
      state[bank][panel].x2 = decode_bf16_device(
          state_input[head_state_base +
                      static_cast<std::size_t>(value0) * kDimension +
                      bank_key1]);
      state[bank][panel].x3 = decode_bf16_device(
          state_input[head_state_base +
                      static_cast<std::size_t>(value1) * kDimension +
                      bank_key1]);
    }
  }
}

// Publish one K64 x V64 update accumulator as the row-major [V64,K64]
// backing required by mma.row.col.  Pair conversion remains packed even
// though the two K rows land eight columns apart in this backing.
__device__ __forceinline__ void faithful_store_state_bank_shared(
    const FaithfulM16N8Accumulator
        (&state)[kFaithfulKeyBanks][kFaithfulValuePanels],
    std::uint16_t* const shared_state,
    const unsigned int bank,
    const unsigned int warp,
    const unsigned int lane) {
  const unsigned int group = lane >> 2U;
  const unsigned int pair = lane & 3U;
  const unsigned int key0 = warp * 16U + group;
  const unsigned int key1 = key0 + 8U;
#pragma unroll
  for (unsigned int panel = 0U; panel < kFaithfulValuePanels;
       ++panel) {
    const unsigned int value0 = panel * 8U + pair * 2U;
    const unsigned int value1 = value0 + 1U;
    const unsigned int chunk00 =
        faithful_swizzled_chunk(value0, key0 / 8U);
    const unsigned int chunk01 =
        faithful_swizzled_chunk(value0, key1 / 8U);
    const unsigned int chunk10 =
        faithful_swizzled_chunk(value1, key0 / 8U);
    const unsigned int chunk11 =
        faithful_swizzled_chunk(value1, key1 / 8U);
    const std::uint32_t packed0 =
        faithful_pack_pair(state[bank][panel].x0,
                           state[bank][panel].x2);
    const std::uint32_t packed1 =
        faithful_pack_pair(state[bank][panel].x1,
                           state[bank][panel].x3);
    shared_state[value0 * kFaithfulPanelColumns + chunk00 * 8U +
                 key0 % 8U] = static_cast<std::uint16_t>(packed0);
    shared_state[value0 * kFaithfulPanelColumns + chunk01 * 8U +
                 key1 % 8U] = static_cast<std::uint16_t>(packed0 >> 16U);
    shared_state[value1 * kFaithfulPanelColumns + chunk10 * 8U +
                 key0 % 8U] = static_cast<std::uint16_t>(packed1);
    shared_state[value1 * kFaithfulPanelColumns + chunk11 * 8U +
                 key1 % 8U] = static_cast<std::uint16_t>(packed1 >> 16U);
  }
}

// The shared backing is already H[V,K], so each mandatory public boundary
// vector is one 128-bit shared load and one 128-bit global store.
__device__ __forceinline__ void faithful_copy_state_bank_global(
    const std::uint16_t* const shared_state,
    std::uint16_t* const global_state,
    const unsigned int value_half,
    const unsigned int bank,
    const unsigned int thread) {
  constexpr unsigned int kVectorCount =
      kFaithfulPanelBytes / sizeof(uint4);
  static_assert(kVectorCount == 512U);
#pragma unroll 1
  for (unsigned int vector = thread; vector < kVectorCount;
       vector += kFusedSolveThreads) {
    const unsigned int value = vector / 8U;
    const unsigned int logical_chunk = vector % 8U;
    const unsigned int key_base = logical_chunk * 8U;
    const unsigned int physical_chunk =
        faithful_swizzled_chunk(value, logical_chunk);
    const uint4 packed = *reinterpret_cast<const uint4*>(
        shared_state + value * kFaithfulPanelColumns +
        physical_chunk * 8U);
    auto* const destination = reinterpret_cast<uint4*>(
        global_state +
        static_cast<std::size_t>(value_half * 64U + value) *
            kDimension +
        bank * 64U + key_base);
    *destination = packed;
  }
}

[[nodiscard]] __device__ __forceinline__ std::uint32_t
faithful_load_shared_pair(const std::uint16_t* const shared_matrix,
                          const unsigned int row,
                          const unsigned int panel,
                          const unsigned int pair) {
  const unsigned int physical_chunk =
      faithful_swizzled_chunk(row, panel);
  return *reinterpret_cast<const std::uint32_t*>(
      shared_matrix + row * kFaithfulPanelColumns +
      physical_chunk * 8U + pair * 2U);
}

__device__ __forceinline__ void faithful_store_v_transposed(
    std::uint16_t* const shared_matrix,
    const unsigned int value,
    const unsigned int token,
    const std::uint16_t rounded) {
  const unsigned int physical_chunk =
      faithful_swizzled_chunk(value, token / 8U);
  shared_matrix[value * kFaithfulPanelColumns +
                physical_chunk * 8U + token % 8U] = rounded;
}

__device__ __forceinline__ void faithful_form_decayed_k_transpose(
    const std::uint16_t* const shared_raw_k0,
    const std::uint16_t* const shared_raw_k1,
    const float* const shared_gamma,
    std::uint16_t* const shared_decayed_k0,
    std::uint16_t* const shared_decayed_k1,
    const unsigned int thread) {
#pragma unroll 1
  for (unsigned int index = thread;
       index < kFaithfulPanelElements;
       index += kFusedSolveThreads) {
    const unsigned int token = index / kFaithfulPanelColumns;
    const unsigned int key = index % kFaithfulPanelColumns;
    const unsigned int raw_chunk =
        faithful_swizzled_chunk(token, key / 8U);
    const unsigned int raw_offset =
        token * kFaithfulPanelColumns + raw_chunk * 8U + key % 8U;
    const unsigned int transposed_chunk =
        faithful_swizzled_chunk(key, token / 8U);
    const unsigned int transposed_offset =
        key * kFaithfulPanelColumns + transposed_chunk * 8U +
        token % 8U;
    const float scale = shared_gamma[token];
    shared_decayed_k0[transposed_offset] = encode_bf16_device(
        decode_bf16_device(shared_raw_k0[raw_offset]) * scale);
    shared_decayed_k1[transposed_offset] = encode_bf16_device(
        decode_bf16_device(shared_raw_k1[raw_offset]) * scale);
  }
}

__device__ __forceinline__ void faithful_scale_state(
    FaithfulM16N8Accumulator
        (&state)[kFaithfulKeyBanks][kFaithfulValuePanels],
    const float decay) {
#pragma unroll
  for (unsigned int bank = 0U; bank < kFaithfulKeyBanks; ++bank) {
#pragma unroll
    for (unsigned int panel = 0U; panel < kFaithfulValuePanels;
         ++panel) {
      state[bank][panel].x0 *= decay;
      state[bank][panel].x1 *= decay;
      state[bank][panel].x2 *= decay;
      state[bank][panel].x3 *= decay;
    }
  }
}

template <bool Diagnostic>
__global__ __launch_bounds__(kFusedSolveThreads, 2)
void persistent_state_chunk64_vllm_faithful_kernel(
    const std::uint16_t* const w,
    const std::uint16_t* const u,
    const std::uint16_t* const compact_k,
    const float* const gamma,
    const std::uint16_t* const state_input,
    std::uint16_t* const state_output,
    const unsigned int chunk_count,
    std::uint16_t* const v_new,
    std::uint16_t* const boundary_state,
    std::uint16_t* const diagnostic_k_decay,
    std::uint16_t* const diagnostic_post_update_state) {
  extern __shared__ __align__(16) unsigned char shared_raw[];
  auto* const shared_w0 = reinterpret_cast<std::uint16_t*>(
      shared_raw + kFaithfulW0Offset);
  auto* const shared_w1 = reinterpret_cast<std::uint16_t*>(
      shared_raw + kFaithfulW1Offset);
  auto* const shared_u = reinterpret_cast<std::uint16_t*>(
      shared_raw + kFaithfulUOffset);
  auto* const shared_k0 = reinterpret_cast<std::uint16_t*>(
      shared_raw + kFaithfulK0Offset);
  auto* const shared_k1 = reinterpret_cast<std::uint16_t*>(
      shared_raw + kFaithfulK1Offset);
  auto* const shared_transient = reinterpret_cast<std::uint16_t*>(
      shared_raw + kFaithfulTransientOffset);
  auto* const shared_gamma = reinterpret_cast<float*>(
      shared_raw + kFaithfulGammaOffset);

  const unsigned int thread = threadIdx.x;
  const unsigned int warp = thread / 32U;
  const unsigned int lane = thread % 32U;
  const unsigned int value_head = blockIdx.x / 2U;
  const unsigned int value_half = blockIdx.x % 2U;
  const std::size_t head_state_base =
      static_cast<std::size_t>(value_head) * kDimension * kDimension;

  FaithfulM16N8Accumulator
      state[kFaithfulKeyBanks][kFaithfulValuePanels];
  faithful_load_initial_state(state, state_input, head_state_base,
                              value_half, warp, lane);

  // One extra publication iteration reuses the exact same BF16 conversion
  // instructions for final state_output instead of cloning a second static
  // conversion tail after the recurrence.
  for (unsigned int chunk_index = 0U;; ++chunk_index) {
    const bool has_chunk = chunk_index < chunk_count;
    const std::size_t matrix =
        static_cast<std::size_t>(chunk_index) * kValueHeadCount +
        value_head;
    const std::size_t compact_matrix =
        static_cast<std::size_t>(chunk_index) * kQkHeadCount +
        value_head / 3U;

    const std::uint16_t* matrix_w = nullptr;
    const std::uint16_t* matrix_u = nullptr;
    const std::uint16_t* matrix_k = nullptr;
    const float* matrix_gamma = nullptr;
    std::uint16_t* matrix_v_new = nullptr;
    std::uint16_t* publication = state_output + head_state_base;
    if (has_chunk) {
      matrix_w = w + matrix * kChunkSize * kDimension;
      matrix_u = u + matrix * kChunkSize * kDimension;
      matrix_k = compact_k +
                 compact_matrix * kChunkSize * kDimension;
      matrix_gamma = gamma + matrix * kChunkSize;
      matrix_v_new = v_new + matrix * kChunkSize * kDimension;
      publication = boundary_state + matrix * kDimension * kDimension;
      faithful_stage_chunk(matrix_w, matrix_u, matrix_k, matrix_gamma,
                           value_half, shared_w0, shared_w1, shared_u,
                           shared_k0, shared_k1, shared_gamma, thread);
    }

    FaithfulM16N8Accumulator products[kFaithfulValuePanels];
#pragma unroll
    for (unsigned int panel = 0U; panel < kFaithfulValuePanels;
         ++panel) {
      faithful_zero(products[panel]);
    }

#pragma unroll
    for (unsigned int bank = 0U; bank < kFaithfulKeyBanks; ++bank) {
      faithful_store_state_bank_shared(state, shared_transient, bank,
                                       warp, lane);
      __syncthreads();
      faithful_copy_state_bank_global(shared_transient, publication,
                                      value_half, bank, thread);
      if (!has_chunk) {
        __syncthreads();
        continue;
      }
      if (bank == 0U) {
        resident_state_cp_async_wait_all();
      }
      __syncthreads();

      const std::uint16_t* const selected_w =
          bank == 0U ? shared_w0 : shared_w1;
#pragma unroll
      for (unsigned int key_pair = 0U; key_pair < 2U; ++key_pair) {
        FaithfulM16K16 w_fragment0{};
        FaithfulM16K16 w_fragment1{};
        faithful_load_a(w_fragment0, selected_w, warp,
                        key_pair * 2U, lane);
        faithful_load_a(w_fragment1, selected_w, warp,
                        key_pair * 2U + 1U, lane);
#pragma unroll
        for (unsigned int panel = 0U;
             panel < kFaithfulValuePanels; ++panel) {
          FaithfulK16N8 state_fragment0{};
          FaithfulK16N8 state_fragment1{};
          faithful_load_b_pair_transposed(
              state_fragment0, state_fragment1, shared_transient,
              panel, key_pair, lane);
          faithful_mma(products[panel], w_fragment0,
                       state_fragment0);
          faithful_mma(products[panel], w_fragment1,
                       state_fragment1);
        }
      }
      __syncthreads();
    }

    if (!has_chunk) {
      return;
    }

    // Products are T16xV64 per warp.  Preserve the public BF16 Vnew boundary
    // and retain that identical rounded tile as row-major [T64,V64] for the
    // update GEMM's matrix-B operand.
    const unsigned int group = lane >> 2U;
    const unsigned int pair = lane & 3U;
    const unsigned int token0 = warp * 16U + group;
    const unsigned int token1 = token0 + 8U;
#pragma unroll
    for (unsigned int panel = 0U; panel < kFaithfulValuePanels;
         ++panel) {
      const std::uint32_t u0 = faithful_load_shared_pair(
          shared_u, token0, panel, pair);
      const std::uint32_t u1 = faithful_load_shared_pair(
          shared_u, token1, panel, pair);
      const float corrected00 =
          decode_bf16_device(static_cast<std::uint16_t>(u0)) -
          products[panel].x0;
      const float corrected01 =
          decode_bf16_device(static_cast<std::uint16_t>(u0 >> 16U)) -
          products[panel].x1;
      const float corrected10 =
          decode_bf16_device(static_cast<std::uint16_t>(u1)) -
          products[panel].x2;
      const float corrected11 =
          decode_bf16_device(static_cast<std::uint16_t>(u1 >> 16U)) -
          products[panel].x3;
      const std::uint32_t rounded0 =
          faithful_pack_pair(corrected00, corrected01);
      const std::uint32_t rounded1 =
          faithful_pack_pair(corrected10, corrected11);
      *reinterpret_cast<std::uint32_t*>(
          matrix_v_new + token0 * kDimension + value_half * 64U +
          panel * 8U + pair * 2U) = rounded0;
      *reinterpret_cast<std::uint32_t*>(
          matrix_v_new + token1 * kDimension + value_half * 64U +
          panel * 8U + pair * 2U) = rounded1;
      const unsigned int value0 = panel * 8U + pair * 2U;
      const unsigned int value1 = value0 + 1U;
      faithful_store_v_transposed(
          shared_transient, value0, token0,
          static_cast<std::uint16_t>(rounded0));
      faithful_store_v_transposed(
          shared_transient, value1, token0,
          static_cast<std::uint16_t>(rounded0 >> 16U));
      faithful_store_v_transposed(
          shared_transient, value0, token1,
          static_cast<std::uint16_t>(rounded1));
      faithful_store_v_transposed(
          shared_transient, value1, token1,
          static_cast<std::uint16_t>(rounded1 >> 16U));
    }
    __syncthreads();

    // The project contract rounds end-decayed K (rather than vLLM's
    // algebraically moved V decay) at BF16.  Keep that boundary exact, but
    // form the transpose in a retained runtime loop so the conversion body
    // appears once in static code.  W is dead here and becomes the K^T slot.
    // U is dead after Vnew. Capture raw g_last there before any warp can
    // overwrite gamma[63] with its unit end-decay scale. Reading gamma[63]
    // directly inside the in-place conversion is a cross-warp race: lane 63
    // may publish 1 before lane 0 has observed the original cumulative gate.
    auto* const shared_raw_last = reinterpret_cast<float*>(shared_u);
    if (thread == 0U) {
      shared_raw_last[0] = shared_gamma[kChunkSize - 1U];
    }
    __syncthreads();
    const float raw_last = shared_raw_last[0];
    if (thread < kChunkSize) {
      shared_gamma[thread] =
          expf(raw_last - shared_gamma[thread]);
      if (thread == 0U) {
        shared_gamma[kChunkSize] = expf(raw_last);
      }
    }
    __syncthreads();
    faithful_form_decayed_k_transpose(
        shared_k0, shared_k1, shared_gamma, shared_w0, shared_w1,
        thread);
    faithful_scale_state(state, shared_gamma[kChunkSize]);
    __syncthreads();

    if constexpr (Diagnostic) {
      // Export the exact shared operand consumed by update MMA in canonical
      // [T,K] order. Only one value-half CTA publishes each value head.
      if (chunk_index == 0U && value_half == 0U) {
        for (unsigned int index = thread;
             index < kChunkSize * kDimension;
             index += kFusedSolveThreads) {
          const unsigned int token = index / kDimension;
          const unsigned int key = index % kDimension;
          const unsigned int bank = key / 64U;
          const unsigned int local_key = key % 64U;
          const std::uint16_t* const selected_k =
              bank == 0U ? shared_w0 : shared_w1;
          const unsigned int physical_chunk =
              faithful_swizzled_chunk(local_key, token / 8U);
          const std::uint16_t value =
              selected_k[local_key * kFaithfulPanelColumns +
                         physical_chunk * 8U + token % 8U];
          diagnostic_k_decay[
              matrix * kChunkSize * kDimension + index] = value;
        }
      }
    }

#pragma unroll
    for (unsigned int bank = 0U; bank < kFaithfulKeyBanks; ++bank) {
      const std::uint16_t* const selected_k =
          bank == 0U ? shared_w0 : shared_w1;
#pragma unroll
      for (unsigned int token_pair = 0U; token_pair < 2U;
           ++token_pair) {
        FaithfulM16K16 k_fragment0{};
        FaithfulM16K16 k_fragment1{};
        faithful_load_a(k_fragment0, selected_k, warp,
                        token_pair * 2U, lane);
        faithful_load_a(k_fragment1, selected_k, warp,
                        token_pair * 2U + 1U, lane);
#pragma unroll
        for (unsigned int panel = 0U;
             panel < kFaithfulValuePanels; ++panel) {
          FaithfulK16N8 v_fragment0{};
          FaithfulK16N8 v_fragment1{};
          faithful_load_b_pair_transposed(
              v_fragment0, v_fragment1, shared_transient, panel,
              token_pair, lane);
          faithful_mma(state[bank][panel], k_fragment0, v_fragment0);
          faithful_mma(state[bank][panel], k_fragment1, v_fragment1);
        }
      }
    }
    __syncthreads();

    if constexpr (Diagnostic) {
      // Materialize the register-resident H immediately after chunk0's
      // update, before the next iteration's boundary publication.
      if (chunk_index == 0U) {
#pragma unroll
        for (unsigned int bank = 0U; bank < kFaithfulKeyBanks; ++bank) {
          faithful_store_state_bank_shared(
              state, shared_transient, bank, warp, lane);
          __syncthreads();
          faithful_copy_state_bank_global(
              shared_transient,
              diagnostic_post_update_state + head_state_base,
              value_half, bank, thread);
          __syncthreads();
        }
      }
    }
  }
}

// Architecture stage C: reconstruct one C64/value-head tile with WMMA, keep
// its FP32 output on chip, then publish the same BF16 boundary consumed by
// plain RMSNorm and SiLU(Z). One CTA therefore replaces the two batched GEMM
// calls plus scale, scatter, and standalone norm/gate materialization.
template <bool CompactQ>
__global__ __launch_bounds__(kThreads)
void reconstruct_norm_gate_chunk64_kernel(
    const std::uint16_t* const boundary_state,
    const std::uint16_t* const q,
    const std::uint16_t* const v_new,
    const std::uint16_t* const qk,
    const float* const gamma,
    const std::uint16_t* const norm_weight,
    const std::uint16_t* const silu_gate,
  const float norm_epsilon,
  std::uint16_t* const output) {
  extern __shared__ float shared_output[];
  const unsigned int warp = threadIdx.x / 32U;
  const unsigned int lane = threadIdx.x % 32U;
  const std::size_t matrix = blockIdx.x;
  const std::size_t chunk_index = matrix / kValueHeadCount;
  const std::size_t value_head = matrix % kValueHeadCount;
  const unsigned int value_base = warp * kSolveSubblock;
  const auto* const matrix_state = reinterpret_cast<const Bf16*>(
      boundary_state + matrix * kDimension * kDimension);
  const std::size_t q_matrix =
      CompactQ ? chunk_index * kQkHeadCount + value_head / 3U : matrix;
  const auto* const matrix_q = reinterpret_cast<const Bf16*>(
      q + q_matrix * kChunkSize * kDimension);
  const auto* const matrix_v_new = reinterpret_cast<const Bf16*>(
      v_new + matrix * kChunkSize * kDimension);
  const auto* const matrix_qk = reinterpret_cast<const Bf16*>(
      qk + matrix * kChunkSize * kChunkSize);

#pragma unroll
  for (unsigned int token_block = 0U;
       token_block < kChunkSize / kSolveSubblock; ++token_block) {
    WmmaAccumulator accumulator;
    wmma::fill_fragment(accumulator, 0.0F);
#pragma unroll
    for (unsigned int key_block = 0U;
         key_block < kDimension / kSolveSubblock; ++key_block) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, Bf16,
                     wmma::row_major>
          state_fragment;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, Bf16,
                     wmma::col_major>
          q_fragment;
      wmma::load_matrix_sync(
          state_fragment,
          matrix_state + value_base * kDimension +
              key_block * kSolveSubblock,
          static_cast<int>(kDimension));
      wmma::load_matrix_sync(
          q_fragment,
          matrix_q + token_block * kSolveSubblock * kDimension +
              key_block * kSolveSubblock,
          static_cast<int>(kDimension));
      wmma::mma_sync(accumulator, state_fragment, q_fragment, accumulator);
    }

    float* const output_tile =
        shared_output + value_base * kChunkSize +
        token_block * kSolveSubblock;
    wmma::store_matrix_sync(output_tile, accumulator,
                            static_cast<int>(kChunkSize),
                            wmma::mem_row_major);
    __syncwarp();
    for (unsigned int index = lane;
         index < kSolveSubblock * kSolveSubblock; index += 32U) {
      const unsigned int row = index / kSolveSubblock;
      const unsigned int column = index % kSolveSubblock;
      const unsigned int token_in_chunk =
          token_block * kSolveSubblock + column;
      output_tile[row * kChunkSize + column] *=
          expf(gamma[matrix * kChunkSize + token_in_chunk]);
    }
    __syncwarp();
    wmma::load_matrix_sync(accumulator, output_tile,
                           static_cast<int>(kChunkSize),
                           wmma::mem_row_major);
#pragma unroll
    for (unsigned int inner_token_block = 0U;
         inner_token_block < kChunkSize / kSolveSubblock;
         ++inner_token_block) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, Bf16,
                     wmma::col_major>
          value_fragment;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, Bf16,
                     wmma::row_major>
          score_fragment;
      wmma::load_matrix_sync(
          value_fragment,
          matrix_v_new +
              inner_token_block * kSolveSubblock * kDimension +
              value_base,
          static_cast<int>(kDimension));
      wmma::load_matrix_sync(
          score_fragment,
          matrix_qk +
              inner_token_block * kSolveSubblock * kChunkSize +
              token_block * kSolveSubblock,
          static_cast<int>(kChunkSize));
      wmma::mma_sync(accumulator, value_fragment, score_fragment,
                     accumulator);
    }
    wmma::store_matrix_sync(output_tile, accumulator,
                            static_cast<int>(kChunkSize),
                            wmma::mem_row_major);
  }
  __syncthreads();

  // Eight warps cover the 64 token rows in eight waves. The recurrence
  // output is rounded to BF16 before the norm exactly as in the prior
  // scatter-then-normalize route.
  for (unsigned int token_in_chunk = warp;
       token_in_chunk < kChunkSize; token_in_chunk += 8U) {
    float square_sum = 0.0F;
#pragma unroll
    for (unsigned int value_dimension = lane;
         value_dimension < kDimension; value_dimension += 32U) {
      const float rounded = decode_bf16_device(encode_bf16_device(
          shared_output[value_dimension * kChunkSize + token_in_chunk]));
      square_sum = fmaf(rounded, rounded, square_sum);
    }
#pragma unroll
    for (unsigned int offset = 16U; offset != 0U; offset >>= 1U) {
      square_sum += __shfl_down_sync(0xffffffffU, square_sum, offset);
    }
    const float inverse_rms = __shfl_sync(
        0xffffffffU,
        rsqrtf(square_sum / static_cast<float>(kDimension) +
               norm_epsilon),
        0U);
    const std::size_t token =
        chunk_index * kChunkSize + token_in_chunk;
#pragma unroll
    for (unsigned int value_dimension = lane;
         value_dimension < kDimension; value_dimension += 32U) {
      const float rounded = decode_bf16_device(encode_bf16_device(
          shared_output[value_dimension * kChunkSize + token_in_chunk]));
      const std::size_t destination =
          token * kGdnVElements + value_head * kDimension +
          value_dimension;
      const float gate = decode_bf16_device(silu_gate[destination]);
      const float normalized =
          rounded * inverse_rms *
          decode_bf16_device(norm_weight[value_dimension]);
      output[destination] = encode_bf16_device(
          normalized * gate / (1.0F + expf(-gate)));
    }
  }
}

[[nodiscard]] int launch_grid_status() noexcept {
  return static_cast<int>(cudaGetLastError());
}

[[nodiscard]] int record_wy_timing_event(
    void* const event, cudaStream_t stream) noexcept {
  if (event == nullptr) {
    return static_cast<int>(cudaSuccess);
  }
  return static_cast<int>(cudaEventRecord(
      reinterpret_cast<cudaEvent_t>(event), stream));
}

[[nodiscard]] bool force_fused_kkt_baseline() noexcept {
  static const bool forced_by_environment =
      std::getenv("Q3X_GDN_CHUNK64_FORCE_FUSED_KKT_BASELINE") != nullptr;
  return forced_by_environment ||
         gdn_prefill_chunk64_native_detail::
             g_force_fused_kkt_baseline_for_test;
}

[[nodiscard]] bool force_resident_state_baseline() noexcept {
  static const bool forced_by_environment =
      std::getenv(
          "Q3X_GDN_CHUNK64_FORCE_RESIDENT_STATE_BASELINE") != nullptr;
  return forced_by_environment ||
         gdn_prefill_chunk64_native_detail::
             g_force_resident_state_baseline_for_test;
}

// Explicit rollback for the incumbent packless resident recurrence. The
// vLLM-faithful recurrence is the native-C64 default after passing the exact
// state, Graph, and warm B-C-C-B promotion gates.
[[nodiscard]] bool force_packless_resident_state_fallback() noexcept {
  static const bool forced_by_environment =
      std::getenv(
          "Q3X_GDN_CHUNK64_FORCE_PACKLESS_RESIDENT_STATE_FALLBACK") !=
      nullptr;
  return forced_by_environment ||
         gdn_prefill_chunk64_native_detail::
             g_force_packless_resident_state_fallback_for_test;
}

// Diagnostic-only selector for a same-ELF direction gate against the exact
// three-launch native producer that immediately preceded group-owned WY.
// Production never sets this environment variable and therefore always uses
// the group-owned route below.
[[nodiscard]] bool force_split_wy_baseline() noexcept {
  static const bool forced_by_environment =
      std::getenv("Q3X_GDN_CHUNK64_FORCE_SPLIT_WY_BASELINE") != nullptr;
  return forced_by_environment ||
         gdn_prefill_chunk64_native_detail::
             g_force_split_wy_baseline_for_test;
}

[[nodiscard]] bool force_packed_qkv_baseline() noexcept {
  static const bool forced_by_environment =
      std::getenv("Q3X_GDN_CHUNK64_FORCE_PACKED_QKV_BASELINE") != nullptr;
  return forced_by_environment ||
         gdn_prefill_chunk64_native_detail::
             g_force_packed_qkv_baseline_for_test;
}

// Same-ELF selector for the admitted value-head-owned C64 hierarchy.  The
// preceding group-owned route remains available for exact and performance
// regression diagnosis, but production takes the admitted route by default.
[[nodiscard]] bool use_vllm_layout_wy_candidate() noexcept {
  const auto route_for_test = gdn_prefill_chunk64_native_detail::
      g_vllm_layout_wy_route_for_test;
  if (route_for_test != gdn_prefill_chunk64_native_detail::
                            VllmLayoutWyRouteForTest::kProductionDefault) {
    return route_for_test == gdn_prefill_chunk64_native_detail::
                                 VllmLayoutWyRouteForTest::kVllmLayout;
  }
  static const bool force_group_owned_baseline =
      std::getenv("Q3X_GDN_CHUNK64_FORCE_GROUP_OWNED_WY_BASELINE") !=
      nullptr;
  return !force_group_owned_baseline;
}

// Promoted production route inside the explicitly admitted native C64
// architecture. The preceding compact-QK plus reconstruct path remains
// available only as a same-ELF diagnostic baseline.
[[nodiscard]] bool use_chunk_o_bv64() noexcept {
  static const bool legacy_baseline_by_environment =
      std::getenv(
          "Q3X_GDN_CHUNK64_FORCE_LEGACY_QK_RECONSTRUCT_BASELINE") != nullptr;
  return !legacy_baseline_by_environment &&
         !gdn_prefill_chunk64_native_detail::
               g_force_legacy_qk_reconstruct_baseline_for_test;
}

[[nodiscard]] bool invalid_arguments(
    void* const /*context*/,
    void* const workspace,
    const std::size_t workspace_capacity_bytes,
    const std::size_t token_count,
    const std::uint16_t* const conv_qkv,
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
  return token_count == 0U || token_count > kTokenCount ||
         token_count % kChunkSize != 0U || workspace == nullptr ||
         workspace_capacity_bytes < required_workspace_bytes() ||
         conv_qkv == nullptr || a == nullptr || b == nullptr ||
         A_log == nullptr || dt_bias == nullptr || state_input == nullptr ||
         state_output == nullptr || norm_weight == nullptr ||
         silu_gate == nullptr || output == nullptr ||
         !std::isfinite(l2_epsilon) || l2_epsilon <= 0.0F ||
         !std::isfinite(norm_epsilon) || norm_epsilon <= 0.0F;
}

}  // namespace

std::size_t workspace_bytes() noexcept { return required_workspace_bytes(); }

int create_context(void** const context) noexcept {
  if (context == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  static int context_token = 0;
  *context = &context_token;
  return static_cast<int>(cudaSuccess);
}

int destroy_context(void* const /*context*/) noexcept {
  return static_cast<int>(cudaSuccess);
}

int launch_impl(void* const context,
                void* const workspace_raw,
                const std::size_t workspace_capacity_bytes,
                const std::size_t token_count,
                const std::uint16_t* const conv_qkv,
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
                void* const cuda_stream,
                const bool qk_preprocessed) noexcept {
  if (invalid_arguments(context, workspace_raw, workspace_capacity_bytes,
                        token_count, conv_qkv, a, b, A_log, dt_bias,
                        state_input, state_output, l2_epsilon, norm_weight,
                        silu_gate, norm_epsilon, output)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  (void)context;
  Workspace workspace;
  if (!partition_workspace(workspace_raw, workspace_capacity_bytes,
                           workspace)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const std::size_t chunk_count = token_count / kChunkSize;
  const std::size_t matrix_count = chunk_count * kValueHeadCount;
  const std::size_t head_token_elements =
      matrix_count * kChunkSize * kDimension;
  int status = static_cast<int>(cudaSuccess);
  (void)cudaGetLastError();

  const bool use_fused_kkt_baseline = force_fused_kkt_baseline();
  const bool use_split_wy_baseline =
      !use_fused_kkt_baseline && force_split_wy_baseline();
  const bool use_resident_state_baseline =
      force_resident_state_baseline();
  const bool use_packless_resident_state_fallback =
      !use_resident_state_baseline &&
      force_packless_resident_state_fallback();
  const bool use_vllm_faithful_state_candidate =
      !use_resident_state_baseline &&
      !use_packless_resident_state_fallback;
  const bool use_packed_qkv_baseline =
      force_packed_qkv_baseline() || use_fused_kkt_baseline ||
      use_split_wy_baseline || use_resident_state_baseline;
  const bool use_chunk_o_bv64_output =
      use_chunk_o_bv64() && !use_packed_qkv_baseline;

  // A preprocessed workspace is layout-compatible only with the production
  // compact route. Diagnostic packed baselines must remain fully self-owned.
  if (qk_preprocessed && use_packed_qkv_baseline) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  if (qk_preprocessed) {
    // The immediately preceding same-stream fused convolution populated the
    // exact compact BF16 Q/K regions.  No host synchronization is required.
  } else if (use_packed_qkv_baseline) {
    normalize_qk_kernel<false><<<
        static_cast<unsigned int>(token_count * kQkHeadCount),
        kNormalizeThreads, 0U, stream>>>(
        conv_qkv, l2_epsilon, workspace.q, workspace.k);
  } else {
    normalize_qk_kernel<true><<<
        static_cast<unsigned int>(token_count * kQkHeadCount),
        kNormalizeThreads, 0U, stream>>>(
        conv_qkv, l2_epsilon, workspace.q, workspace.k);
  }
  status = launch_grid_status();
  if (status != static_cast<int>(cudaSuccess)) {
    return status;
  }
  const std::size_t compact_qk_elements =
      chunk_count * kQkHeadCount * kChunkSize * kDimension;
  gdn_prefill_chunk64_native_detail::inspect_preprocess_boundaries(
      workspace.q, compact_qk_elements, workspace.k,
      compact_qk_elements, cuda_stream);
  prepare_gate_kernel<<<static_cast<unsigned int>(matrix_count),
                        kChunkThreads, 0U, stream>>>(
      a, b, A_log, dt_bias, workspace.gamma, workspace.beta);
  status = launch_grid_status();
  if (status != static_cast<int>(cudaSuccess)) {
    return status;
  }
  if (use_packed_qkv_baseline) {
    const unsigned int head_token_blocks = static_cast<unsigned int>(
        (head_token_elements + kThreads - 1U) / kThreads);
    pack_scaled_k_v_kernel<<<head_token_blocks, kThreads, 0U, stream>>>(
        conv_qkv, workspace.k, workspace.gamma, head_token_elements,
        workspace.k_g, workspace.k_decay, workspace.v);
    status = launch_grid_status();
    if (status != static_cast<int>(cudaSuccess)) {
      return status;
    }
  }

  const auto timing_hook =
      gdn_prefill_chunk64_native_detail::g_wy_timing_hook;
  status = record_wy_timing_event(timing_hook.begin, stream);
  if (status != static_cast<int>(cudaSuccess)) {
    return status;
  }
  if (use_fused_kkt_baseline) {
    constexpr std::size_t fused_solve_shared_bytes =
        kChunkSize * kDimension * sizeof(Bf16) +
        2U * kChunkSize * kChunkSize * sizeof(float);
    fused_kkt_solve_block16_kernel<<<
        static_cast<unsigned int>(matrix_count), kFusedSolveThreads,
        fused_solve_shared_bytes, stream>>>(
        workspace.k, workspace.gamma, workspace.beta,
        workspace.transform);
    status = launch_grid_status();
    if (status != static_cast<int>(cudaSuccess)) {
      return status;
    }
  } else if (use_split_wy_baseline) {
    streamed_kkt_gate_chunk64_kernel<<<
        static_cast<unsigned int>(matrix_count), kStreamedKktThreads,
        kStreamedKktSharedBytes, stream>>>(
        workspace.k, workspace.gamma, workspace.beta, workspace.kkt);
    status = launch_grid_status();
    if (status != static_cast<int>(cudaSuccess)) {
      return status;
    }
    solve_kkt_block16_register_kernel<<<
        static_cast<unsigned int>(matrix_count), kFusedSolveThreads,
        kSplitSolveSharedBytes, stream>>>(
        workspace.kkt, workspace.beta, workspace.transform);
    status = launch_grid_status();
    if (status != static_cast<int>(cudaSuccess)) {
      return status;
    }
  } else if (use_packed_qkv_baseline) {
    status = gdn_prefill_group_wy_detail::launch(
        workspace.k, workspace.gamma, workspace.beta, workspace.k_g,
        workspace.v, chunk_count, workspace.transform, workspace.w,
        workspace.u, cuda_stream);
    if (status != static_cast<int>(cudaSuccess)) {
      return status;
    }
  } else if (use_vllm_layout_wy_candidate()) {
    status = gdn_prefill_wy_vllm_layout_detail::launch_packless(
        workspace.k, workspace.gamma, workspace.beta, conv_qkv,
        chunk_count, workspace.kkt, workspace.transform, workspace.w,
        workspace.u, cuda_stream);
    if (status != static_cast<int>(cudaSuccess)) {
      return status;
    }
  } else {
    status = gdn_prefill_group_wy_detail::launch_packless(
        workspace.k, workspace.gamma, workspace.beta, conv_qkv,
        chunk_count, workspace.transform, workspace.w, workspace.u,
        cuda_stream);
    if (status != static_cast<int>(cudaSuccess)) {
      return status;
    }
  }
  status = record_wy_timing_event(timing_hook.after_initial, stream);
  if (status != static_cast<int>(cudaSuccess)) {
    return status;
  }

  if (!use_chunk_o_bv64_output) {
    if (use_packed_qkv_baseline) {
      qk_scaled_chunk64_kernel<<<static_cast<unsigned int>(matrix_count),
                                 kQkThreads, kQkSharedBytes, stream>>>(
          workspace.q, workspace.k, workspace.gamma, workspace.qk);
    } else {
      qk_scaled_group_chunk64_kernel<<<
          static_cast<unsigned int>(chunk_count * kQkHeadCount),
          kQkThreads, kQkSharedBytes, stream>>>(
          workspace.q, workspace.k, workspace.gamma, workspace.qk);
    }
    status = launch_grid_status();
    if (status != static_cast<int>(cudaSuccess)) {
      return status;
    }
  }
  status = record_wy_timing_event(timing_hook.after_qk, stream);
  if (status != static_cast<int>(cudaSuccess)) {
    return status;
  }

  if (use_fused_kkt_baseline || use_split_wy_baseline) {
    recompute_w_u_chunk64_kernel<<<static_cast<unsigned int>(matrix_count),
                                   kWuThreads, kWuSharedBytes, stream>>>(
        workspace.transform, workspace.k_g, workspace.v, workspace.w,
        workspace.u);
    status = launch_grid_status();
    if (status != static_cast<int>(cudaSuccess)) {
      return status;
    }
  }
  status = record_wy_timing_event(timing_hook.after_final, stream);
  if (status != static_cast<int>(cudaSuccess)) {
    return status;
  }

  if (use_resident_state_baseline) {
    constexpr std::size_t persistent_state_baseline_shared_bytes =
        64U * kDimension * (sizeof(float) + sizeof(Bf16));
    persistent_state_chunk64_baseline_kernel<<<
        static_cast<unsigned int>(kValueHeadCount * 2U),
        kFusedSolveThreads, persistent_state_baseline_shared_bytes, stream>>>(
        workspace.w, workspace.u, workspace.k_decay, workspace.gamma,
        state_input, state_output, static_cast<unsigned int>(chunk_count),
        workspace.v_new, workspace.boundary_state);
  } else if (use_packed_qkv_baseline) {
    persistent_state_chunk64_resident_kernel<false><<<
        static_cast<unsigned int>(kValueHeadCount * 2U),
        kFusedSolveThreads, kResidentStateSharedBytes, stream>>>(
        workspace.w, workspace.u, workspace.k_decay, workspace.gamma,
        state_input, state_output, static_cast<unsigned int>(chunk_count),
        workspace.v_new, workspace.boundary_state);
  } else if (use_vllm_faithful_state_candidate) {
    // The runner invokes this launcher once per GDN layer.  Opt in to the
    // 49,412-byte dynamic shared lifetime once during candidate warmup,
    // rather than adding 48 driver calls to every measured prefix.
    const bool diagnostic =
        gdn_prefill_chunk64_native_detail::g_inspection_hook.callback !=
        nullptr;
    if (diagnostic) {
      static const int attribute_status = static_cast<int>(
          cudaFuncSetAttribute(
              persistent_state_chunk64_vllm_faithful_kernel<true>,
              cudaFuncAttributeMaxDynamicSharedMemorySize,
              static_cast<int>(kFaithfulStateSharedBytes)));
      status = attribute_status;
      if (status != static_cast<int>(cudaSuccess)) {
        return status;
      }
      persistent_state_chunk64_vllm_faithful_kernel<true><<<
          static_cast<unsigned int>(kValueHeadCount * 2U),
          kFusedSolveThreads, kFaithfulStateSharedBytes, stream>>>(
          workspace.w, workspace.u, workspace.k, workspace.gamma,
          state_input, state_output,
          static_cast<unsigned int>(chunk_count), workspace.v_new,
          workspace.boundary_state, workspace.k_decay, workspace.k_g);
    } else {
      static const int attribute_status = static_cast<int>(
          cudaFuncSetAttribute(
              persistent_state_chunk64_vllm_faithful_kernel<false>,
              cudaFuncAttributeMaxDynamicSharedMemorySize,
              static_cast<int>(kFaithfulStateSharedBytes)));
      status = attribute_status;
      if (status != static_cast<int>(cudaSuccess)) {
        return status;
      }
      persistent_state_chunk64_vllm_faithful_kernel<false><<<
          static_cast<unsigned int>(kValueHeadCount * 2U),
          kFusedSolveThreads, kFaithfulStateSharedBytes, stream>>>(
          workspace.w, workspace.u, workspace.k, workspace.gamma,
          state_input, state_output,
          static_cast<unsigned int>(chunk_count), workspace.v_new,
          workspace.boundary_state, nullptr, nullptr);
    }
  } else {
    persistent_state_chunk64_resident_kernel<true><<<
        static_cast<unsigned int>(kValueHeadCount * 2U),
        kFusedSolveThreads, kResidentStatePacklessSharedBytes, stream>>>(
        workspace.w, workspace.u, workspace.k, workspace.gamma,
        state_input, state_output, static_cast<unsigned int>(chunk_count),
        workspace.v_new, workspace.boundary_state);
  }
  status = launch_grid_status();
  if (status != static_cast<int>(cudaSuccess)) {
    return status;
  }

  if (use_chunk_o_bv64_output) {
    // Packless WY/state never touch workspace.v. Reuse that exact-size
    // legacy region for the BF16 [T,H,V] boundary between BV64 owners and
    // the independent exact D128 norm/gate epilogue.
    status = gdn_prefill_chunk_o_bv64_detail::launch(
        workspace.q, workspace.k, workspace.boundary_state,
        workspace.v_new, workspace.gamma, token_count, norm_weight,
        silu_gate, norm_epsilon, workspace.v, output, cuda_stream);
  } else {
    constexpr std::size_t reconstruction_shared_bytes =
        kDimension * kChunkSize * sizeof(float);
    if (use_packed_qkv_baseline) {
      reconstruct_norm_gate_chunk64_kernel<false><<<
          static_cast<unsigned int>(matrix_count), kThreads,
          reconstruction_shared_bytes, stream>>>(
          workspace.boundary_state, workspace.q, workspace.v_new,
          workspace.qk, workspace.gamma, norm_weight, silu_gate,
          norm_epsilon, output);
    } else {
      reconstruct_norm_gate_chunk64_kernel<true><<<
          static_cast<unsigned int>(matrix_count), kThreads,
          reconstruction_shared_bytes, stream>>>(
          workspace.boundary_state, workspace.q, workspace.v_new,
          workspace.qk, workspace.gamma, norm_weight, silu_gate,
          norm_epsilon, output);
    }
    status = launch_grid_status();
  }
  if (status == static_cast<int>(cudaSuccess)) {
    gdn_prefill_chunk64_native_detail::inspect_native_boundaries(
        workspace.transform, matrix_count * kChunkSize * kChunkSize,
        workspace.w, head_token_elements, workspace.u,
        head_token_elements, workspace.v_new, head_token_elements,
        workspace.boundary_state,
        matrix_count * kDimension * kDimension, workspace.k,
        chunk_count * kQkHeadCount * kChunkSize * kDimension,
        workspace.gamma, matrix_count * kChunkSize,
        workspace.k_decay, head_token_elements, workspace.k_g,
        kGdnStateElements, state_output, kGdnStateElements, output,
        token_count * kGdnVElements, cuda_stream);
  }
  return status;
}

int launch(void* const context,
           void* const workspace_raw,
           const std::size_t workspace_capacity_bytes,
           const std::size_t token_count,
           const std::uint16_t* const conv_qkv,
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
  return launch_impl(context, workspace_raw, workspace_capacity_bytes,
                     token_count, conv_qkv, a, b, A_log, dt_bias,
                     state_input, state_output, l2_epsilon, norm_weight,
                     silu_gate, norm_epsilon, output, cuda_stream, false);
}

int query_native_resources(int* const registers_per_thread,
                           std::size_t* const static_shared_bytes,
                           std::size_t* const local_bytes,
                           int* const maximum_threads_per_block,
                           int* const active_blocks_per_sm) noexcept {
  if (registers_per_thread == nullptr || static_shared_bytes == nullptr ||
      local_bytes == nullptr || maximum_threads_per_block == nullptr ||
      active_blocks_per_sm == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (use_chunk_o_bv64()) {
    return gdn_prefill_chunk_o_bv64_detail::query_chunk_o_resources(
        registers_per_thread, static_shared_bytes, local_bytes,
        maximum_threads_per_block, active_blocks_per_sm);
  }
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes, persistent_state_chunk64_resident_kernel<true>);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active, persistent_state_chunk64_resident_kernel<true>,
      static_cast<int>(kFusedSolveThreads),
      kResidentStatePacklessSharedBytes);
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

}  // namespace q3x::runtime::gdn_prefill_chunk64_reference_detail

namespace q3x::runtime::gdn_prefill_chunk64_native_detail {

std::size_t workspace_bytes() noexcept {
  return gdn_prefill_chunk64_reference_detail::workspace_bytes();
}

int launch(void* const workspace,
           const std::size_t workspace_capacity_bytes,
           const std::uint16_t* const conv_qkv,
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
  if (token_count == 0U || token_count > 512U || token_count % 64U != 0U) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  return gdn_prefill_chunk64_reference_detail::launch(
      nullptr, workspace, workspace_capacity_bytes, token_count, conv_qkv, a,
      b, A_log, dt_bias, state_input, state_output, l2_epsilon, norm_weight,
      silu_gate, norm_epsilon, output, cuda_stream);
}

int launch_fused_conv_compact_qk_preprocess(
    void* const workspace_raw,
    const std::size_t workspace_capacity_bytes,
    const std::uint16_t* const raw_qkv,
    const std::size_t token_count,
    const std::uint16_t* const conv_weight,
    std::uint16_t* const history_in_out,
    std::uint16_t* const conv_qkv_output,
    const float l2_epsilon,
    void* const cuda_stream) noexcept {
  gdn_prefill_chunk64_reference_detail::Workspace workspace;
  if (token_count == 0U || token_count > 512U || token_count % 64U != 0U ||
      !gdn_prefill_chunk64_reference_detail::partition_workspace(
          workspace_raw, workspace_capacity_bytes, workspace)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  return gdn_prefill_whole_span_conv_detail::
      launch_causal_conv1d_silu_update_token_parallel_compact_qk_exact_cuda(
          raw_qkv, token_count, conv_weight, history_in_out,
          conv_qkv_output, l2_epsilon, workspace.q, workspace.k,
          cuda_stream);
}

int launch_qk_preprocessed(
    void* const workspace,
    const std::size_t workspace_capacity_bytes,
    const std::uint16_t* const conv_qkv,
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
  if (token_count == 0U || token_count > 512U || token_count % 64U != 0U) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  return gdn_prefill_chunk64_reference_detail::launch_impl(
      nullptr, workspace, workspace_capacity_bytes, token_count, conv_qkv, a,
      b, A_log, dt_bias, state_input, state_output, l2_epsilon, norm_weight,
      silu_gate, norm_epsilon, output, cuda_stream, true);
}

int launch_compact_qk_baseline_for_test(
    const std::uint16_t* const conv_qkv,
    const std::size_t token_count,
    const float l2_epsilon,
    std::uint16_t* const compact_q,
    std::uint16_t* const compact_k,
    void* const cuda_stream) noexcept {
  if (conv_qkv == nullptr || compact_q == nullptr || compact_k == nullptr ||
      compact_q == compact_k || compact_q == conv_qkv ||
      compact_k == conv_qkv || token_count == 0U || token_count > 512U ||
      token_count % 64U != 0U || !std::isfinite(l2_epsilon) ||
      l2_epsilon <= 0.0F) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  gdn_prefill_chunk64_reference_detail::normalize_qk_kernel<true><<<
      static_cast<unsigned int>(
          token_count *
          gdn_prefill_chunk64_reference_detail::kQkHeadCount),
      gdn_prefill_chunk64_reference_detail::kNormalizeThreads, 0U,
      stream>>>(conv_qkv, l2_epsilon, compact_q, compact_k);
  return gdn_prefill_chunk64_reference_detail::launch_grid_status();
}

const void* compact_qk_baseline_kernel_handle_for_test() noexcept {
  return reinterpret_cast<const void*>(
      gdn_prefill_chunk64_reference_detail::normalize_qk_kernel<true>);
}

int query_resources(int* const registers_per_thread,
                    std::size_t* const static_shared_bytes,
                    std::size_t* const local_bytes,
                    int* const maximum_threads_per_block,
                    int* const active_blocks_per_sm) noexcept {
  return gdn_prefill_chunk64_reference_detail::query_native_resources(
      registers_per_thread, static_shared_bytes, local_bytes,
      maximum_threads_per_block, active_blocks_per_sm);
}

}  // namespace q3x::runtime::gdn_prefill_chunk64_native_detail
