#include "q3x/kernels/sm87_phase_local_weight_expansion.h"

#include "q3x/quantization/nvfp4.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <type_traits>

namespace {

using q3x::kernels::Sm87PhaseLocalWeightExpansionCapability;
using q3x::kernels::Sm87PhaseLocalWeightExpansionPlan;
using q3x::kernels::Sm87PhaseLocalWeightExpansionResources;
using q3x::kernels::Sm87PhaseLocalWeightFormat;
using q3x::kernels::Sm87PhaseLocalWeightRole;
using q3x::kernels::sm87_phase_local_weight_expansion_plan;

constexpr Sm87PhaseLocalWeightExpansionCapability kCapability{
    8, 7, true, true, true};

constexpr auto kNvGate = sm87_phase_local_weight_expansion_plan(
    Sm87PhaseLocalWeightRole::kNvFp4Gate, kCapability);
constexpr auto kNvUp = sm87_phase_local_weight_expansion_plan(
    Sm87PhaseLocalWeightRole::kNvFp4Up, kCapability);
constexpr auto kNvDown = sm87_phase_local_weight_expansion_plan(
    Sm87PhaseLocalWeightRole::kNvFp4Down, kCapability);
constexpr auto kLinearQkv = sm87_phase_local_weight_expansion_plan(
    Sm87PhaseLocalWeightRole::kFp8LinearQkv, kCapability);
constexpr auto kLinearZ = sm87_phase_local_weight_expansion_plan(
    Sm87PhaseLocalWeightRole::kFp8LinearZ, kCapability);
constexpr auto kLinearOutput = sm87_phase_local_weight_expansion_plan(
    Sm87PhaseLocalWeightRole::kFp8LinearOutput, kCapability);
constexpr auto kFullQuery = sm87_phase_local_weight_expansion_plan(
    Sm87PhaseLocalWeightRole::kFp8FullQuery, kCapability);
constexpr auto kFullKey = sm87_phase_local_weight_expansion_plan(
    Sm87PhaseLocalWeightRole::kFp8FullKey, kCapability);
constexpr auto kFullValue = sm87_phase_local_weight_expansion_plan(
    Sm87PhaseLocalWeightRole::kFp8FullValue, kCapability);
constexpr auto kFullOutput = sm87_phase_local_weight_expansion_plan(
    Sm87PhaseLocalWeightRole::kFp8FullOutput, kCapability);

static_assert(kCapability.valid());
static_assert(kNvGate.valid() && kNvUp.valid() && kNvDown.valid());
static_assert(kLinearQkv.valid() && kLinearZ.valid() &&
              kLinearOutput.valid() && kFullQuery.valid() &&
              kFullKey.valid() && kFullValue.valid() &&
              kFullOutput.valid());

static_assert(kNvGate.format ==
              Sm87PhaseLocalWeightFormat::kNvFp4ModelOptBlock16);
static_assert(kNvGate.output_features == 17'408U &&
              kNvGate.input_features == 5'120U);
static_assert(kNvGate.canonical_weight_bytes == 44'564'480U);
static_assert(kNvGate.block_scale_bytes == 5'570'560U);
static_assert(kNvGate.scratch_elements == 89'128'960U);
static_assert(kNvGate.scratch_bytes == 178'257'920U);
static_assert(kNvUp.canonical_weight_bytes ==
              kNvGate.canonical_weight_bytes);
static_assert(kNvDown.output_features == 5'120U &&
              kNvDown.input_features == 17'408U);
static_assert(kNvDown.canonical_weight_bytes == 44'564'480U);
static_assert(kNvDown.block_scale_bytes == 5'570'560U);
static_assert(kNvDown.scratch_bytes == 178'257'920U);

static_assert(kLinearQkv.format == Sm87PhaseLocalWeightFormat::kFp8E4M3Fn);
static_assert(kLinearQkv.canonical_weight_bytes == 52'428'800U &&
              kLinearQkv.block_scale_bytes == 0U &&
              kLinearQkv.scratch_bytes == 104'857'600U);
static_assert(kLinearZ.canonical_weight_bytes == 31'457'280U &&
              kLinearZ.scratch_bytes == 62'914'560U);
static_assert(kLinearOutput.canonical_weight_bytes == 31'457'280U &&
              kLinearOutput.scratch_bytes == 62'914'560U);
static_assert(kFullQuery.canonical_weight_bytes == 62'914'560U &&
              kFullQuery.scratch_bytes == 125'829'120U);
static_assert(kFullKey.canonical_weight_bytes == 5'242'880U &&
              kFullKey.scratch_bytes == 10'485'760U);
static_assert(kFullValue.canonical_weight_bytes == 5'242'880U &&
              kFullValue.scratch_bytes == 10'485'760U);
static_assert(kFullOutput.canonical_weight_bytes == 31'457'280U &&
              kFullOutput.scratch_bytes == 62'914'560U);

static_assert(kNvGate.fp32_global_scale_bytes == sizeof(float));
static_assert(kLinearQkv.fp32_global_scale_bytes == sizeof(float));
static_assert(!kNvGate.global_scale_baked &&
              !kLinearQkv.global_scale_baked);
static_assert(kNvGate.canonical_row_major && kNvGate.phase_local &&
              kNvGate.test_only);
static_assert(kNvGate.blocks == kNvGate.output_features &&
              kNvGate.threads == 256U);

constexpr bool invalid_capabilities_fail_closed() {
  auto wrong_major = kCapability;
  wrong_major.compute_major = 9;
  auto wrong_minor = kCapability;
  wrong_minor.compute_minor = 0;
  auto no_opt_in = kCapability;
  no_opt_in.test_only_opt_in = false;
  auto wrong_layout = kCapability;
  wrong_layout.canonical_checkpoint_layout = false;
  auto persistent = kCapability;
  persistent.phase_local_scratch = false;
  return !Sm87PhaseLocalWeightExpansionCapability{}.valid() &&
         !wrong_major.valid() && !wrong_minor.valid() && !no_opt_in.valid() &&
         !wrong_layout.valid() && !persistent.valid() &&
         !sm87_phase_local_weight_expansion_plan(
              Sm87PhaseLocalWeightRole::kNvFp4Gate, no_opt_in)
              .valid() &&
         !sm87_phase_local_weight_expansion_plan(
              static_cast<Sm87PhaseLocalWeightRole>(255U), kCapability)
              .valid();
}

constexpr bool forged_plans_fail_closed() {
  auto wrong_format = kNvGate;
  wrong_format.format = Sm87PhaseLocalWeightFormat::kFp8E4M3Fn;
  auto wrong_k = kNvDown;
  wrong_k.input_features = 5'120U;
  auto wrong_weight_span = kLinearQkv;
  ++wrong_weight_span.canonical_weight_bytes;
  auto wrong_scale_span = kNvGate;
  --wrong_scale_span.block_scale_bytes;
  auto wrong_global_span = kNvGate;
  wrong_global_span.fp32_global_scale_bytes = 2U;
  auto wrong_scratch = kFullQuery;
  wrong_scratch.scratch_bytes -= 2U;
  auto persistent = kNvGate;
  persistent.phase_local = false;
  auto production = kNvGate;
  production.test_only = false;
  auto baked = kNvGate;
  baked.global_scale_baked = true;
  return !Sm87PhaseLocalWeightExpansionPlan{}.valid() &&
         !wrong_format.valid() && !wrong_k.valid() &&
         !wrong_weight_span.valid() && !wrong_scale_span.valid() &&
         !wrong_global_span.valid() && !wrong_scratch.valid() &&
         !persistent.valid() && !production.valid() && !baked.valid();
}

static_assert(invalid_capabilities_fail_closed());
static_assert(forged_plans_fail_closed());

constexpr Sm87PhaseLocalWeightExpansionResources kValidResourceReceipt{
    Sm87PhaseLocalWeightRole::kNvFp4Gate, 8, 7, 87, 32,
    0U, 0U, 0U, 6, true};
constexpr auto kUnsupportedRoleResourceReceipt = [] {
  auto resources = kValidResourceReceipt;
  resources.role = static_cast<Sm87PhaseLocalWeightRole>(255U);
  resources.registers_per_thread = 0;
  return resources;
}();
static_assert(kValidResourceReceipt.valid());
static_assert(!kUnsupportedRoleResourceReceipt.valid());

using NvFp4Launcher = int (*)(
    const Sm87PhaseLocalWeightExpansionPlan&, const std::uint8_t*,
    const std::uint8_t*, std::uint16_t*, std::size_t, void*) noexcept;
using Fp8Launcher = int (*)(
    const Sm87PhaseLocalWeightExpansionPlan&, const std::uint8_t*,
    std::uint16_t*, std::size_t, void*) noexcept;
static_assert(std::is_same_v<
              decltype(&q3x::kernels::
                           launch_sm87_phase_local_nvfp4_weight_expansion_test_cuda),
              NvFp4Launcher>);
static_assert(std::is_same_v<
              decltype(&q3x::kernels::
                           launch_sm87_phase_local_fp8_weight_expansion_test_cuda),
              Fp8Launcher>);

[[nodiscard]] std::uint16_t bf16_rne_bits(const float value) noexcept {
  std::uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  if ((bits & 0x7fff'ffffU) > 0x7f80'0000U) {
    return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
  }
  return static_cast<std::uint16_t>(
      (bits + 0x0000'7fffU + ((bits >> 16U) & 1U)) >> 16U);
}

[[nodiscard]] std::uint32_t float_bits(const float value) noexcept {
  std::uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

[[nodiscard]] float bf16_bits_to_float(const std::uint16_t value) noexcept {
  const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16U;
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

[[nodiscard]] bool check(const bool condition, const char* const message) {
  if (!condition) {
    std::cerr << message << '\n';
  }
  return condition;
}

[[nodiscard]] bool exhaustive_fp8_decode_matches() {
  for (unsigned int code = 0U; code < 256U; ++code) {
    const auto raw = static_cast<std::uint8_t>(code);
    const std::uint16_t expected =
        bf16_rne_bits(q3x::quantization::decode_e4m3fn(raw));
    const std::uint16_t actual =
        q3x::kernels::sm87_phase_local_fp8_expanded_bf16_reference(raw);
    if (actual != expected) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool exhaustive_nvfp4_decode_matches() {
  for (unsigned int nibble = 0U; nibble < 16U; ++nibble) {
    for (unsigned int scale = 0U; scale < 256U; ++scale) {
      const auto weight_code = static_cast<std::uint8_t>(nibble);
      const auto scale_code = static_cast<std::uint8_t>(scale);
      const std::uint8_t packed_low = weight_code;
      const std::uint8_t packed_high =
          static_cast<std::uint8_t>(weight_code << 4U);
      const float expected_value =
          q3x::quantization::decode_e2m1(weight_code) *
          q3x::quantization::decode_e4m3fn(scale_code);
      const std::uint16_t expected = bf16_rne_bits(expected_value);
      const std::uint16_t actual_low =
          q3x::kernels::sm87_phase_local_nvfp4_expanded_bf16_reference(
              packed_low, false, scale_code);
      const std::uint16_t actual_high =
          q3x::kernels::sm87_phase_local_nvfp4_expanded_bf16_reference(
              packed_high, true, scale_code);
      if (actual_low != expected || actual_high != expected) {
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] bool exhaustive_finite_nvfp4_products_are_exact_bf16() {
  std::size_t finite_scale_count = 0U;
  std::size_t finite_product_count = 0U;
  for (unsigned int scale = 0U; scale < 256U; ++scale) {
    const float scale_value = q3x::quantization::decode_e4m3fn(
        static_cast<std::uint8_t>(scale));
    if (!std::isfinite(scale_value)) {
      continue;
    }
    ++finite_scale_count;
    for (unsigned int nibble = 0U; nibble < 16U; ++nibble) {
      const float product = q3x::quantization::decode_e2m1(
                                static_cast<std::uint8_t>(nibble)) *
                            scale_value;
      const float bf16_product = bf16_bits_to_float(bf16_rne_bits(product));
      if (float_bits(product) != float_bits(bf16_product)) {
        return false;
      }
      ++finite_product_count;
    }
  }
  return finite_scale_count == 254U && finite_product_count == 4'064U &&
         !std::isfinite(q3x::quantization::decode_e4m3fn(0x7fU)) &&
         !std::isfinite(q3x::quantization::decode_e4m3fn(0xffU));
}

[[nodiscard]] bool global_scale_is_not_baked() {
  constexpr std::uint8_t kOneE2M1 = 0x02U;
  constexpr std::uint8_t kOneE4M3 = 0x38U;
  constexpr float kGlobalScale = 0.25F;
  const std::uint16_t expanded =
      q3x::kernels::sm87_phase_local_nvfp4_expanded_bf16_reference(
          kOneE2M1, false, kOneE4M3);
  const std::uint16_t block_local = bf16_rne_bits(
      q3x::quantization::decode_e2m1(kOneE2M1) *
      q3x::quantization::decode_e4m3fn(kOneE4M3));
  const std::uint16_t fully_scaled = bf16_rne_bits(
      q3x::quantization::dequantize_nvfp4_value(
          kOneE2M1, false, kOneE4M3, kGlobalScale));
  return expanded == block_local && expanded != fully_scaled;
}

}  // namespace

int main() {
  bool ok = true;
  ok &= check(exhaustive_fp8_decode_matches(),
              "all 256 E4M3FN codes must match the audited decoder");
  ok &= check(exhaustive_nvfp4_decode_matches(),
              "all 16 E2M1 codes x 256 E4M3FN scales must match");
  ok &= check(exhaustive_finite_nvfp4_products_are_exact_bf16(),
              "every finite E2M1 x E4M3FN product must be exactly BF16");
  ok &= check(global_scale_is_not_baked(),
              "phase-local expansion must preserve the FP32 global scale");
  ok &= check(q3x::kernels::
                      sm87_phase_local_nvfp4_expanded_bf16_reference(
                          0x08U, false, 0x38U) == 0x8000U,
              "ModelOpt E2M1 negative zero must survive expansion");

  // Force linker coverage of the CUDA entry points without launching them.
  volatile NvFp4Launcher nvfp4_launcher =
      &q3x::kernels::
          launch_sm87_phase_local_nvfp4_weight_expansion_test_cuda;
  volatile Fp8Launcher fp8_launcher =
      &q3x::kernels::
          launch_sm87_phase_local_fp8_weight_expansion_test_cuda;
  ok &= check(nvfp4_launcher != nullptr && fp8_launcher != nullptr,
              "test-only CUDA expansion entry points must link");
  return ok ? 0 : 1;
}
