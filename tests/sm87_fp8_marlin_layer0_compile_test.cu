#include "q3x/kernels/sm87_fp8_marlin_w8a16.h"
#include "q3x/kernels/sm87_nvfp4_marlin.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
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
static_assert(!kernels::sm87_fp8_marlin_supports_token_count(513U));
static_assert(!kernels::sm87_fp8_marlin_supports_token_count(8'192U));
static_assert(!kernels::sm87_fp8_marlin_supports_token_count(8'193U));
static_assert(
    kernels::sm87_fp8_marlin_supports_operator_panel_token_count(513U));
static_assert(
    kernels::sm87_fp8_marlin_supports_operator_panel_token_count(8'192U));
static_assert(
    !kernels::sm87_fp8_marlin_supports_operator_panel_token_count(8'193U));
static_assert(kernels::sm87_fp8_marlin_execution_plan(407U).primary_tokens ==
              384U);
static_assert(kernels::sm87_fp8_marlin_execution_plan(481U).remainder_tokens ==
              33U);
static_assert(kernels::sm87_fp8_marlin_tile_config(33U, 1'024U).thread_m ==
              48U);
static_assert(kernels::sm87_fp8_marlin_tile_config(
                  33U, kLayer0Qkv.output_size)
                  .thread_m == 48U);
static_assert(kernels::sm87_fp8_marlin_tile_config(
                  8'192U, kLayer0Qkv.output_size)
                  .thread_m == 64U);
static_assert(kernels::sm87_fp8_marlin_execution_plan(
                  8'192U, kLayer0Qkv.output_size)
                  .launch_count == 8U);
static_assert(kernels::sm87_fp8_marlin_execution_plan(8'192U, 1'024U)
                  .launch_count == 1U);
static_assert(kernels::sm87_fp8_marlin_execution_plan(
                  8'191U, kLayer0Qkv.output_size)
                  .launch_count == 9U);
static_assert(kernels::sm87_fp8_marlin_execution_plan(8'191U, 1'024U)
                  .launch_count == 2U);

static_assert(kernels::kSm87NvFp4MarlinTokens == 512U);
static_assert(kernels::kSm87NvFp4MarlinM512CompatibilityTokens == 512U);
static_assert(!kernels::sm87_nvfp4_marlin_supports_token_count(513U));
static_assert(!kernels::sm87_nvfp4_marlin_supports_token_count(8'192U));
static_assert(!kernels::sm87_nvfp4_marlin_supports_token_count(8'193U));
static_assert(
    kernels::sm87_nvfp4_marlin_supports_operator_panel_token_count(513U));
static_assert(
    kernels::sm87_nvfp4_marlin_supports_operator_panel_token_count(8'192U));
static_assert(
    !kernels::sm87_nvfp4_marlin_supports_operator_panel_token_count(8'193U));
static_assert(
    kernels::sm87_nvfp4_marlin_execution_plan(513U).launch_count == 2U);
static_assert(
    kernels::sm87_nvfp4_marlin_execution_plan(8'192U).launch_count == 8U);
static_assert(
    kernels::sm87_nvfp4_marlin_execution_plan(8'191U).launch_count == 9U);
static_assert(kernels::sm87_nvfp4_marlin_tile_config(8'192U).thread_m == 64U);

[[nodiscard]] bool validate_nvfp4_plan(const std::size_t token_count) {
  const kernels::Sm87NvFp4MarlinExecutionPlan plan =
      kernels::sm87_nvfp4_marlin_execution_plan(token_count);
  std::size_t expected_offset = 0U;
  for (std::size_t index = 0U; index < plan.launch_count; ++index) {
    const kernels::Sm87NvFp4MarlinLaunchSegment segment =
        kernels::sm87_nvfp4_marlin_launch_segment(plan, index);
    if (!segment.valid() || segment.token_offset != expected_offset ||
        segment.token_count >
            kernels::kSm87NvFp4MarlinMaximumKernelSegmentTokens) {
      return false;
    }
    expected_offset += segment.token_count;
  }
  return expected_offset == token_count &&
         !kernels::sm87_nvfp4_marlin_launch_segment(plan, plan.launch_count)
              .valid();
}

[[nodiscard]] bool validate_fp8_plan(const std::size_t token_count,
                                     const std::size_t output_size) {
  const kernels::Sm87Fp8MarlinExecutionPlan plan =
      kernels::sm87_fp8_marlin_execution_plan(token_count, output_size);
  std::size_t expected_offset = 0U;
  for (std::size_t index = 0U; index < plan.launch_count; ++index) {
    const kernels::Sm87Fp8MarlinLaunchSegment segment =
        kernels::sm87_fp8_marlin_launch_segment(plan, index);
    if (!segment.valid() || segment.token_offset != expected_offset ||
        segment.token_count > plan.maximum_segment_tokens) {
      return false;
    }
    expected_offset += segment.token_count;
  }
  return expected_offset == token_count &&
         !kernels::sm87_fp8_marlin_launch_segment(plan, plan.launch_count)
              .valid();
}

}  // namespace

int main() {
  for (std::size_t token_count = 1U;
       token_count <=
       kernels::kSm87NvFp4MarlinMaximumOperatorPanelTokens;
       ++token_count) {
    if (!validate_nvfp4_plan(token_count)) {
      std::cerr << "NVFP4_MARLIN_PANEL_CONTRACT_FAIL tokens=" << token_count
                << '\n';
      return 1;
    }
  }
  for (std::size_t token_count = 2U;
       token_count <= kernels::kSm87Fp8MarlinMaximumOperatorPanelTokens;
       ++token_count) {
    if (!validate_fp8_plan(token_count, kLayer0Qkv.output_size) ||
        !validate_fp8_plan(token_count, 1'024U)) {
      std::cerr << "FP8_MARLIN_PANEL_CONTRACT_FAIL tokens=" << token_count
                << '\n';
      return 1;
    }
  }
  if (kernels::sm87_nvfp4_marlin_execution_plan(0U).launch_count != 0U ||
      kernels::sm87_nvfp4_marlin_execution_plan(8'193U).launch_count != 0U ||
      kernels::sm87_fp8_marlin_execution_plan(1U, 1'024U).launch_count != 0U ||
      kernels::sm87_fp8_marlin_execution_plan(8'193U, 1'024U).launch_count !=
          0U) {
    std::cerr << "MARLIN_PANEL_REJECTION_CONTRACT_FAIL\n";
    return 1;
  }

  const int prepare_status =
      kernels::prepare_sm87_fp8_marlin_projection_cuda(
          nullptr, nullptr, kLayer0Qkv.output_size, kLayer0Qkv.input_size,
          nullptr, nullptr, nullptr, 0U, nullptr);
  const int launch_status = kernels::launch_sm87_fp8_marlin_projection_cuda(
      nullptr, nullptr, nullptr, kernels::kSm87Fp8MarlinC32Tokens,
      kLayer0Qkv.output_size, kLayer0Qkv.input_size, nullptr, nullptr, nullptr,
      nullptr);
  const int maximum_panel_launch_status =
      kernels::launch_sm87_fp8_marlin_projection_cuda(
          nullptr, nullptr, nullptr,
          kernels::kSm87Fp8MarlinMaximumOperatorPanelTokens,
          kLayer0Qkv.output_size, kLayer0Qkv.input_size, nullptr, nullptr,
          nullptr, nullptr);
  const int nvfp4_maximum_panel_launch_status =
      kernels::launch_sm87_nvfp4_marlin_gate_up_cuda(
          nullptr, nullptr, nullptr, nullptr,
          kernels::kSm87NvFp4MarlinMaximumOperatorPanelTokens, nullptr,
          nullptr, nullptr, nullptr);
  if (prepare_status != static_cast<int>(cudaErrorInvalidValue) ||
      launch_status != static_cast<int>(cudaErrorInvalidValue) ||
      maximum_panel_launch_status != static_cast<int>(cudaErrorInvalidValue) ||
      nvfp4_maximum_panel_launch_status !=
          static_cast<int>(cudaErrorInvalidValue)) {
    std::cerr << "FP8_MARLIN_COMPILE_CONTRACT_FAIL prepare=" << prepare_status
              << " launch=" << launch_status
              << " maximum_panel_launch=" << maximum_panel_launch_status
              << " nvfp4_maximum_panel_launch="
              << nvfp4_maximum_panel_launch_status << '\n';
    return 1;
  }
  std::cout << "FP8_MARLIN_COMPILE_CONTRACT_PASS frozen_vllm="
               "ccd49f6821ee110cc5a2b1aba620a8a1d66c7cbb"
            << " layer0_qkv_n=" << kLayer0Qkv.output_size
            << " layer0_qkv_k=" << kLayer0Qkv.input_size
            << " legacy_admission_max=M512 candidate_scheduler=M2..M8192"
               " large_n_segment=M1024"
               " small_n_segment=M8192\n";
  return 0;
}
