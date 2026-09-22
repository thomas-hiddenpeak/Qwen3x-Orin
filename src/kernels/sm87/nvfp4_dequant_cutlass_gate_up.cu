#include "q3x/kernels/sm87_nvfp4_dequant_cutlass_gate_up.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/thread/linear_combination.h"

namespace q3x::kernels {
namespace {

// E2M1 (FP4) decode.
__device__ __forceinline__ float decode_e2m1_device(const std::uint8_t nib) {
  static const __device__ float table[16] = {
      0.0F,  0.5F,  1.0F,  1.5F,  2.0F,  3.0F,  4.0F,  6.0F,
      -0.0F, -0.5F, -1.0F, -1.5F, -2.0F, -3.0F, -4.0F, -6.0F};
  return table[nib & 0x0fU];
}

// E4M3 (FP8) decode.
__device__ __forceinline__ float decode_e4m3fn_device(const std::uint8_t bits) {
  const std::uint8_t magnitude = bits & 0x7fU;
  const int exponent = static_cast<int>((magnitude >> 3U) & 0x0fU);
  const int mantissa = static_cast<int>(magnitude & 0x07U);
  if (exponent == 0x0f && mantissa == 0x07) {
    return nanf("");
  }
  const float value = exponent == 0
                          ? ldexpf(static_cast<float>(mantissa), -9)
                          : ldexpf(1.0F + static_cast<float>(mantissa) / 8.0F,
                                   exponent - 7);
  return (bits & 0x80U) != 0U ? -value : value;
}

// Vectorized NVFP4 dequant: 16 elements/thread.
__global__ void dequant_nvfp4_vec16_kernel(
    const std::uint8_t* const packed, const std::uint8_t* const block_scale,
    const float weight_scale_2, std::uint16_t* const out,
    const long n, const long k) {
  const long groups_per_row = k / 16L;
  const long total_groups = n * groups_per_row;
  const long gidx =
      static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (gidx >= total_groups) {
    return;
  }
  const long row = gidx / groups_per_row;
  const long col = (gidx % groups_per_row) * 16L;

  const std::uint32_t* const pptr =
      reinterpret_cast<const std::uint32_t*>(&packed[row * (k / 2) + col / 2]);
  const std::uint32_t p0 = pptr[0];
  const std::uint32_t p1 = pptr[1];
  const float bs =
      decode_e4m3fn_device(block_scale[row * (k / 16) + col / 16]) *
      weight_scale_2;

  std::uint16_t vals[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const std::uint32_t word = (i < 8) ? p0 : p1;
    const int lane = (i < 8) ? i : (i - 8);
    const std::uint8_t nib = static_cast<std::uint8_t>((word >> (4 * lane)) & 0x0fU);
    const float v = decode_e2m1_device(nib) * bs;
    vals[i] = __bfloat16_as_ushort(__float2bfloat16_rn(v));
  }
  uint4* const optr =
      reinterpret_cast<uint4*>(&out[row * k + col]);
  optr[0] = *reinterpret_cast<uint4*>(&vals[0]);
  optr[1] = *reinterpret_cast<uint4*>(&vals[8]);
}

// SiLU-mul: activated[m,n] = silu(gate) * up
__global__ void silu_mul_kernel(
    const std::uint16_t* const gate_up_out, std::uint16_t* const activated,
    const long m, const long n) {
  constexpr long kVec = 8L;
  const long nv = n / kVec;
  const long total = m * nv;
  const long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  const long m_idx = idx / nv;
  const long v = idx % nv;
  const long n_idx = v * kVec;
  const long g_off = m_idx * 2L * n + n_idx;
  const long u_off = m_idx * 2L * n + n + n_idx;
  const uint4 g4 = *reinterpret_cast<const uint4*>(&gate_up_out[g_off]);
  const uint4 u4 = *reinterpret_cast<const uint4*>(&gate_up_out[u_off]);
  const __nv_bfloat16* const gb =
      reinterpret_cast<const __nv_bfloat16*>(&g4);
  const __nv_bfloat16* const ub =
      reinterpret_cast<const __nv_bfloat16*>(&u4);
  std::uint16_t out[kVec];
#pragma unroll
  for (long i = 0; i < kVec; ++i) {
    const float g = __bfloat162float(gb[i]);
    const float u = __bfloat162float(ub[i]);
    const float silu_g = g / (1.0F + expf(-g));
    out[i] = __bfloat16_as_ushort(__float2bfloat16_rn(silu_g * u));
  }
  *reinterpret_cast<uint4*>(&activated[m_idx * n + n_idx]) =
      *reinterpret_cast<uint4*>(out);
}

// CUTLASS 2.x GEMM: 31.3 TF @ M=8000 vs cuBLAS 29.9 TF (+4.7%, stable over 3 runs).
// Tile 128x256x64, warp 64x64, 3 stages, swizzle 2 (sw2 beats sw1/sw4 on this shape).
using CutlassGemm = cutlass::gemm::device::Gemm<
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    cutlass::bfloat16_t, cutlass::layout::ColumnMajor,
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
        cutlass::bfloat16_t, 8, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<2>,
    3>;

std::uint16_t* g_dequant_buffer = nullptr;
std::size_t g_dequant_capacity = 0U;
std::uint16_t* g_gemm_temp = nullptr;
std::size_t g_gemm_temp_capacity = 0U;

constexpr std::size_t kMChunk = 8000U;

[[nodiscard]] int ensure_buffers(const std::size_t dequant_elements,
                                 const std::size_t gemm_elements) noexcept {
  if (g_dequant_buffer == nullptr || g_dequant_capacity < dequant_elements) {
    std::uint16_t* buf = nullptr;
    const cudaError_t s =
        cudaMalloc(&buf, dequant_elements * sizeof(std::uint16_t));
    if (s != cudaSuccess) {
      return static_cast<int>(s);
    }
    if (g_dequant_buffer != nullptr) {
      cudaFree(g_dequant_buffer);
    }
    g_dequant_buffer = buf;
    g_dequant_capacity = dequant_elements;
  }
  if (g_gemm_temp == nullptr || g_gemm_temp_capacity < gemm_elements) {
    std::uint16_t* buf = nullptr;
    const cudaError_t s =
        cudaMalloc(&buf, gemm_elements * sizeof(std::uint16_t));
    if (s != cudaSuccess) {
      return static_cast<int>(s);
    }
    if (g_gemm_temp != nullptr) {
      cudaFree(g_gemm_temp);
    }
    g_gemm_temp = buf;
    g_gemm_temp_capacity = gemm_elements;
  }
  return static_cast<int>(cudaSuccess);
}

}  // namespace

int launch_nvfp4_dequant_cutlass_gate_up(
    const std::uint8_t* const gate_packed,
    const std::uint8_t* const gate_scale, const float gate_ws2,
    const std::uint8_t* const up_packed, const std::uint8_t* const up_scale,
    const float up_ws2, const std::uint16_t* const input,
    std::uint16_t* const activated, const std::size_t m, const std::size_t n,
    const std::size_t k, void* const cuda_stream) noexcept {
  if (gate_packed == nullptr || gate_scale == nullptr ||
      up_packed == nullptr || up_scale == nullptr || input == nullptr ||
      activated == nullptr || m == 0U || n == 0U || k == 0U ||
      cuda_stream == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const std::size_t dequant_elements = 2U * n * k;
  const std::size_t gemm_elements = kMChunk * 2U * n;
  const int buf_status = ensure_buffers(dequant_elements, gemm_elements);
  if (buf_status != static_cast<int>(cudaSuccess)) {
    return buf_status;
  }
  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  constexpr unsigned int kThreads = 256U;

  // Dequant gate: [n, k] -> g_dequant_buffer[0 : n*k]
  {
    const long total_groups = static_cast<long>(n) * (k / 16L);
    const unsigned int blocks =
        static_cast<unsigned int>((total_groups + kThreads - 1U) / kThreads);
    dequant_nvfp4_vec16_kernel<<<blocks, kThreads, 0U, stream>>>(
        gate_packed, gate_scale, gate_ws2, g_dequant_buffer,
        static_cast<long>(n), static_cast<long>(k));
  }
  // Dequant up: [n, k] -> g_dequant_buffer[n*k : 2*n*k]
  {
    const long total_groups = static_cast<long>(n) * (k / 16L);
    const unsigned int blocks =
        static_cast<unsigned int>((total_groups + kThreads - 1U) / kThreads);
    dequant_nvfp4_vec16_kernel<<<blocks, kThreads, 0U, stream>>>(
        up_packed, up_scale, up_ws2, g_dequant_buffer + n * k,
        static_cast<long>(n), static_cast<long>(k));
  }
  const cudaError_t dq_status = cudaGetLastError();
  if (dq_status != cudaSuccess) {
    return static_cast<int>(dq_status);
  }

  // M-chunked CUTLASS GEMM (sw2): out[mc, 2n] = input[mc, k] * W[2n, k]^T
  const float alpha = 1.0F;
  const float beta = 0.0F;
  for (std::size_t mc = 0; mc < m; mc += kMChunk) {
    const std::size_t mc_size = (m - mc < kMChunk) ? (m - mc) : kMChunk;
    const int M_ = static_cast<int>(mc_size);
    const int N_ = static_cast<int>(2U * n);
    const int K_ = static_cast<int>(k);

    CutlassGemm::Arguments args{
        cutlass::gemm::GemmCoord(M_, N_, K_),
        {reinterpret_cast<cutlass::bfloat16_t const*>(input + mc * k), K_},
        {reinterpret_cast<cutlass::bfloat16_t const*>(g_dequant_buffer), K_},
        {reinterpret_cast<cutlass::bfloat16_t const*>(g_gemm_temp), N_},
        {reinterpret_cast<cutlass::bfloat16_t*>(g_gemm_temp), N_},
        {alpha, beta},
        1
    };
    CutlassGemm gemm;
    const size_t ws_size = CutlassGemm::get_workspace_size(args);
    void* ws = nullptr;
    if (ws_size) {
      const cudaError_t ws_err = cudaMalloc(&ws, ws_size);
      if (ws_err != cudaSuccess) {
        return static_cast<int>(ws_err);
      }
    }
    const cutlass::Status init_status = gemm.initialize(args, ws);
    if (init_status != cutlass::Status::kSuccess) {
      if (ws) cudaFree(ws);
      return static_cast<int>(cudaErrorInvalidValue);
    }
    const cutlass::Status run_status = gemm.run(stream);
    if (ws) cudaFree(ws);
    if (run_status != cutlass::Status::kSuccess) {
      return static_cast<int>(cudaErrorInvalidValue);
    }

    // SiLU-mul: activated[mc : mc+mc_size, n] = silu(gate) * up
    {
      const long total = static_cast<long>(mc_size) * (n / 8L);
      const unsigned int blocks =
          static_cast<unsigned int>((total + kThreads - 1U) / kThreads);
      silu_mul_kernel<<<blocks, kThreads, 0U, stream>>>(
          g_gemm_temp, activated + mc * n, static_cast<long>(mc_size),
          static_cast<long>(n));
    }
  }
  return static_cast<int>(cudaPeekAtLastError());
}

}  // namespace q3x::kernels
