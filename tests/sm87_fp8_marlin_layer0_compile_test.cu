#include "q3x/kernels/sm87_fp8_marlin_w8a16.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <iostream>

namespace {

namespace kernels = q3x::kernels;

constexpr kernels::Sm87Fp8MarlinShape kLayer0Qkv =
    kernels::sm87_fp8_marlin_shape(
        kernels::Sm87Fp8MarlinProjection::kLinearQkv);
constexpr kernels::Sm87Fp8MarlinShape kLayer0Z =
    kernels::sm87_fp8_marlin_shape(
        kernels::Sm87Fp8MarlinProjection::kLinearZ);

static_assert(kLayer0Qkv.output_size == 10'240U &&
              kLayer0Qkv.input_size == 5'120U);
static_assert(kLayer0Z.output_size == 6'144U &&
              kLayer0Z.input_size == 5'120U);
static_assert(kernels::sm87_fp8_marlin_weight_bytes(
                  kLayer0Qkv.output_size, kLayer0Qkv.input_size) ==
              52'428'800U);
static_assert(kernels::sm87_fp8_marlin_scale_bytes(
                  kLayer0Qkv.output_size) == 20'480U);
static_assert(kernels::sm87_fp8_marlin_supports_token_count(2U));
static_assert(kernels::sm87_fp8_marlin_supports_token_count(407U));
static_assert(kernels::sm87_fp8_marlin_execution_plan(407U).primary_tokens ==
              384U);
static_assert(kernels::sm87_fp8_marlin_execution_plan(481U).remainder_tokens ==
              33U);
static_assert(kernels::sm87_fp8_marlin_tile_config(33U, 1'024U).thread_m ==
              48U);
static_assert(kernels::sm87_fp8_marlin_tile_config(
                  33U, kLayer0Qkv.output_size)
                  .thread_m == 48U);

}  // namespace

int main() {
  const int prepare_status =
      kernels::prepare_sm87_fp8_marlin_projection_cuda(
          nullptr, nullptr, kLayer0Qkv.output_size, kLayer0Qkv.input_size,
          nullptr, nullptr, nullptr, 0U, nullptr);
  const int launch_status = kernels::launch_sm87_fp8_marlin_projection_cuda(
      nullptr, nullptr, nullptr, kernels::kSm87Fp8MarlinC32Tokens,
      kLayer0Qkv.output_size, kLayer0Qkv.input_size, nullptr, nullptr, nullptr,
      nullptr);
  if (prepare_status != static_cast<int>(cudaErrorInvalidValue) ||
      launch_status != static_cast<int>(cudaErrorInvalidValue)) {
    std::cerr << "FP8_MARLIN_COMPILE_CONTRACT_FAIL prepare=" << prepare_status
              << " launch=" << launch_status << '\n';
    return 1;
  }
  std::cout << "FP8_MARLIN_COMPILE_CONTRACT_PASS frozen_vllm="
               "ccd49f6821ee110cc5a2b1aba620a8a1d66c7cbb"
            << " layer0_qkv_n=" << kLayer0Qkv.output_size
            << " layer0_qkv_k=" << kLayer0Qkv.input_size
            << " scheduler=M2..M512\n";
  return 0;
}
