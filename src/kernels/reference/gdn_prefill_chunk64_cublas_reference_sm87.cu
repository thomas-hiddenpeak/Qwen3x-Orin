#include "gdn_prefill_chunk64_cublas_reference_sm87.h"
#include "../sm87/gdn_prefill_chunk64_native_sm87.h"

#include "q3x/runtime/gdn_decode.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

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
constexpr std::size_t kHeadTokenElements =
    kMatrixCount * kChunkSize * kDimension;
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
  // Q, K, exp(g)K, end-decayed K, V, T, QK, W, U, Vnew, and H.
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
  // Cumulative gate and beta. QK is produced directly at its BF16 stage
  // boundary by the native WMMA kernel, so no FP32 global scratch is needed.
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

// Architecture stage B: one CTA owns 64 value rows of one head and keeps the
// 64x128 FP32 state tile in eight WMMA accumulator fragments per warp across
// all eight chunks. The shared banks only materialize the BF16 boundary used
// by W and by output reconstruction; no intermediate state is written back
// and reloaded between chunks.
__global__ __launch_bounds__(kFusedSolveThreads)
void persistent_state_chunk64_kernel(
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

// Architecture stage C: reconstruct one C64/value-head tile with WMMA, keep
// its FP32 output on chip, then publish the same BF16 boundary consumed by
// plain RMSNorm and SiLU(Z). One CTA therefore replaces the two batched GEMM
// calls plus scale, scatter, and standalone norm/gate materialization.
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
  const auto* const matrix_q = reinterpret_cast<const Bf16*>(
      q + matrix * kChunkSize * kDimension);
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

  normalize_qk_kernel<<<
      static_cast<unsigned int>(token_count * kQkHeadCount),
      kNormalizeThreads, 0U, stream>>>(conv_qkv, l2_epsilon, workspace.q,
                                       workspace.k);
  status = launch_grid_status();
  if (status != static_cast<int>(cudaSuccess)) {
    return status;
  }
  prepare_gate_kernel<<<static_cast<unsigned int>(matrix_count),
                        kChunkThreads, 0U, stream>>>(
      a, b, A_log, dt_bias, workspace.gamma, workspace.beta);
  status = launch_grid_status();
  if (status != static_cast<int>(cudaSuccess)) {
    return status;
  }
  const unsigned int head_token_blocks = static_cast<unsigned int>(
      (head_token_elements + kThreads - 1U) / kThreads);
  pack_scaled_k_v_kernel<<<head_token_blocks, kThreads, 0U, stream>>>(
      conv_qkv, workspace.k, workspace.gamma, head_token_elements,
      workspace.k_g,
      workspace.k_decay, workspace.v);
  status = launch_grid_status();
  if (status != static_cast<int>(cudaSuccess)) {
    return status;
  }

  constexpr std::size_t fused_solve_shared_bytes =
      kChunkSize * kDimension * sizeof(Bf16) +
      2U * kChunkSize * kChunkSize * sizeof(float);
  fused_kkt_solve_block16_kernel<<<
      static_cast<unsigned int>(matrix_count), kFusedSolveThreads,
      fused_solve_shared_bytes, stream>>>(
      workspace.k, workspace.gamma, workspace.beta, workspace.transform);
  status = launch_grid_status();
  if (status != static_cast<int>(cudaSuccess)) {
    return status;
  }

  qk_scaled_chunk64_kernel<<<static_cast<unsigned int>(matrix_count),
                             kQkThreads, kQkSharedBytes, stream>>>(
      workspace.q, workspace.k, workspace.gamma, workspace.qk);
  status = launch_grid_status();
  if (status != static_cast<int>(cudaSuccess)) {
    return status;
  }

  recompute_w_u_chunk64_kernel<<<static_cast<unsigned int>(matrix_count),
                                 kWuThreads, kWuSharedBytes, stream>>>(
      workspace.transform, workspace.k_g, workspace.v, workspace.w,
      workspace.u);
  status = launch_grid_status();
  if (status != static_cast<int>(cudaSuccess)) {
    return status;
  }

  constexpr std::size_t persistent_state_shared_bytes =
      64U * kDimension * (sizeof(float) + sizeof(Bf16));
  persistent_state_chunk64_kernel<<<
      static_cast<unsigned int>(kValueHeadCount * 2U),
      kFusedSolveThreads, persistent_state_shared_bytes, stream>>>(
      workspace.w, workspace.u, workspace.k_decay, workspace.gamma,
      state_input, state_output, static_cast<unsigned int>(chunk_count),
      workspace.v_new,
      workspace.boundary_state);
  status = launch_grid_status();
  if (status != static_cast<int>(cudaSuccess)) {
    return status;
  }

  constexpr std::size_t reconstruction_shared_bytes =
      kDimension * kChunkSize * sizeof(float);
  reconstruct_norm_gate_chunk64_kernel<<<
      static_cast<unsigned int>(matrix_count), kThreads,
      reconstruction_shared_bytes, stream>>>(
      workspace.boundary_state, workspace.q, workspace.v_new,
      workspace.qk, workspace.gamma, norm_weight, silu_gate,
      norm_epsilon, output);
  return launch_grid_status();
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
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes, recompute_w_u_chunk64_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active, recompute_w_u_chunk64_kernel, static_cast<int>(kWuThreads),
      kWuSharedBytes);
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
