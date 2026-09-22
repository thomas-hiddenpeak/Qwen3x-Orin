#include "q3x/kernels/sm87_fp8_dequant_cublas_projection.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/thread/linear_combination.h"

#include <cstdint>

namespace q3x::kernels {
namespace {

constexpr std::size_t kMaxFatNWeightElements = 12'288U * 5'120U;  // fa_q

// Canonical FP8 E4M3 -> float. Matches src/kernels/reference/fp8_reference.cu.
__device__ __forceinline__ float decode_e4m3fn_device(const std::uint8_t bits) {
  const std::uint8_t magnitude = bits & 0x7fU;
  const int exponent = static_cast<int>((magnitude >> 3U) & 0x0fU);
  const int mantissa = static_cast<int>(magnitude & 0x07U);
  if (exponent == 0x0f && mantissa == 0x07) {
    return nanf("");
  }
  float value = exponent == 0
                    ? ldexpf(static_cast<float>(mantissa), -9)
                    : ldexpf(1.0F + static_cast<float>(mantissa) / 8.0F,
                             exponent - 7);
  return (bits & 0x80U) != 0U ? -value : value;
}

__global__ void dequant_fp8_to_bf16_kernel(
    const std::uint8_t* const fp8_weight, const float weight_scale,
    std::uint16_t* const out_bf16, const std::size_t total) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }
  const float value =
      decode_e4m3fn_device(fp8_weight[index]) * weight_scale;
  out_bf16[index] = __bfloat16_as_ushort(__float2bfloat16_rn(value));
}

// CUTLASS 2.x GEMM (Ampere cp.async): 128x256x64, warp 64x64, 3 stages, swizzle 2.
// Measured at M=8000 K=5120: N=10240 -> 37.3 TF vs cuBLAS 29.7 TF (+25%),
// N=12288 -> 35.8 TF vs cuBLAS 30.3 TF (+18%). Stable over 3 runs.
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
std::size_t g_dequant_buffer_capacity = 0U;

[[nodiscard]] int ensure_dequant_buffer(const std::size_t elements) noexcept {
  if (g_dequant_buffer != nullptr && g_dequant_buffer_capacity >= elements) {
    return static_cast<int>(cudaSuccess);
  }
  const std::size_t capacity =
      elements > kMaxFatNWeightElements ? elements : kMaxFatNWeightElements;
  std::uint16_t* buffer = nullptr;
  const cudaError_t status =
      cudaMalloc(&buffer, capacity * sizeof(std::uint16_t));
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  g_dequant_buffer = buffer;
  g_dequant_buffer_capacity = capacity;
  return static_cast<int>(cudaSuccess);
}

}  // namespace

int launch_fp8_dequant_cublas_projection(
    const std::uint8_t* const fp8_weight, const float weight_scale,
    const std::uint16_t* const input_bf16, std::uint16_t* const output_bf16,
    const std::size_t m, const std::size_t n, const std::size_t k,
    void* const cuda_stream) noexcept {
  if (fp8_weight == nullptr || input_bf16 == nullptr || output_bf16 == nullptr ||
      m == 0U || n == 0U || k == 0U || cuda_stream == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const std::size_t total = n * k;
  const int buffer_status = ensure_dequant_buffer(total);
  if (buffer_status != static_cast<int>(cudaSuccess)) {
    return buffer_status;
  }
  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  constexpr unsigned int kThreads = 256U;
  const std::size_t blocks = (total + kThreads - 1U) / kThreads;
  dequant_fp8_to_bf16_kernel<<<static_cast<unsigned int>(blocks), kThreads,
                               0U, stream>>>(fp8_weight, weight_scale,
                                              g_dequant_buffer, total);
  const cudaError_t dequant_status = cudaGetLastError();
  if (dequant_status != cudaSuccess) {
    return static_cast<int>(dequant_status);
  }
  // out[M,N] row-major = input[M,K] row-major * W[N,K]^T.
  // CUTLASS: A=input (M,K) RowMajor, B=W (K,N) ColumnMajor [stored (N,K)
  // row-major], C=output (M,N) RowMajor.
  const float alpha = 1.0F;
  const float beta = 0.0F;
  CutlassGemm::Arguments args{
      cutlass::gemm::GemmCoord(static_cast<int>(m), static_cast<int>(n),
                               static_cast<int>(k)),
      {reinterpret_cast<cutlass::bfloat16_t const*>(input_bf16),
       static_cast<int>(k)},
      {reinterpret_cast<cutlass::bfloat16_t const*>(g_dequant_buffer),
       static_cast<int>(k)},
      {reinterpret_cast<cutlass::bfloat16_t const*>(output_bf16),
       static_cast<int>(n)},
      {reinterpret_cast<cutlass::bfloat16_t*>(output_bf16),
       static_cast<int>(n)},
      {alpha, beta},
      1
  };
  CutlassGemm gemm;
  const size_t ws_size = CutlassGemm::get_workspace_size(args);
  void* ws = nullptr;
  if (ws_size) {
    const cudaError_t ws_status = cudaMalloc(&ws, ws_size);
    if (ws_status != cudaSuccess) {
      return static_cast<int>(ws_status);
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
  return static_cast<int>(cudaSuccess);
}

}  // namespace q3x::kernels
