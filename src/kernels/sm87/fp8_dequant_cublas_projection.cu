#include "q3x/kernels/sm87_fp8_dequant_cublas_projection.h"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

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

std::uint16_t* g_dequant_buffer = nullptr;
std::size_t g_dequant_buffer_capacity = 0U;
cublasHandle_t g_cublas_handle = nullptr;

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

[[nodiscard]] int ensure_cublas_handle() noexcept {
  if (g_cublas_handle != nullptr) {
    return static_cast<int>(cudaSuccess);
  }
  const cublasStatus_t status = cublasCreate(&g_cublas_handle);
  return status == CUBLAS_STATUS_SUCCESS ? static_cast<int>(cudaSuccess)
                                         : static_cast<int>(cudaErrorMemoryAllocation);
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
  const int handle_status = ensure_cublas_handle();
  if (handle_status != static_cast<int>(cudaSuccess)) {
    return handle_status;
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
  // cuBLAS column-major: C^T[N,M] = W^T[N,K](opT) * input^T[K,M](opN).
  if (cublasSetStream(g_cublas_handle, stream) != CUBLAS_STATUS_SUCCESS) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const float alpha = 1.0F;
  const float beta = 0.0F;
  const cublasStatus_t gemm_status = cublasGemmEx(
      g_cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
      static_cast<int>(n), static_cast<int>(m), static_cast<int>(k),
      &alpha, g_dequant_buffer, CUDA_R_16BF, static_cast<int>(k),
      input_bf16, CUDA_R_16BF, static_cast<int>(k), &beta, output_bf16,
      CUDA_R_16BF, static_cast<int>(n), CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT);
  if (gemm_status != CUBLAS_STATUS_SUCCESS) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  return static_cast<int>(cudaSuccess);
}

}  // namespace q3x::kernels
