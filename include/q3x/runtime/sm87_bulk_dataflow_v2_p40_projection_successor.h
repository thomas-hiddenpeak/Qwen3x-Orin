#pragma once

#include "q3x/kernels/sm87_bulk_dataflow_v2_fp8_projection.h"
#include "q3x/kernels/sm87_bulk_dataflow_v2_fp8_whole_p40.h"
#include "q3x/kernels/sm87_bulk_dataflow_v2_nvfp4_down_whole_p40.h"
#include "q3x/kernels/sm87_bulk_dataflow_v2_nvfp4_gate_up_whole_p40.h"
#include "q3x/kernels/sm87_bulk_dataflow_v2_nvfp4_projection.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace q3x::runtime {

// Host-only topology contract for the three whole-P40000 projection
// successors in AC-PREFILL-SM87-BULK-DATAFLOW-v2.  It deliberately separates
// checkpoint roles, fused outer operations, and physical CUDA launches.  The
// old FP8 M1024/M64 family and NVFP4 M256/M1024 exact-control family remain
// useful same-ELF numerical oracles, but their segmented launch counts can
// never satisfy this receipt.
inline constexpr std::array<std::uint8_t, 8U>
    kSm87BulkV2P40ProjectionSuccessorMagic{
        {'Q', '3', 'X', 'P', 'S', 'V', '2', '1'}};
inline constexpr std::uint16_t
    kSm87BulkV2P40ProjectionSuccessorAbiMajor = 1U;
inline constexpr std::uint16_t
    kSm87BulkV2P40ProjectionSuccessorAbiMinor = 0U;

inline constexpr std::size_t kSm87BulkV2P40ProjectionTokens = 40'000U;

inline constexpr std::size_t kSm87BulkV2P40Fp8GdnInputLogicalRoles = 96U;
inline constexpr std::size_t kSm87BulkV2P40Fp8FullInputLogicalRoles = 48U;
inline constexpr std::size_t kSm87BulkV2P40Fp8OutputLogicalRoles = 64U;
inline constexpr std::size_t kSm87BulkV2P40Fp8LogicalRoles = 208U;
inline constexpr std::size_t kSm87BulkV2P40Fp8GdnInputOuterOperations = 48U;
inline constexpr std::size_t kSm87BulkV2P40Fp8FullInputOuterOperations = 16U;
inline constexpr std::size_t kSm87BulkV2P40Fp8OutputOuterOperations = 64U;
inline constexpr std::size_t kSm87BulkV2P40Fp8FusedOuterOperations = 128U;
inline constexpr std::size_t kSm87BulkV2P40Fp8WholeRoleLaunches = 128U;

inline constexpr std::size_t kSm87BulkV2P40NvFp4GateUpLogicalRoles = 128U;
inline constexpr std::size_t kSm87BulkV2P40NvFp4DownLogicalRoles = 64U;
inline constexpr std::size_t kSm87BulkV2P40NvFp4LogicalRoles = 192U;
inline constexpr std::size_t kSm87BulkV2P40NvFp4GateUpOuterOperations = 64U;
inline constexpr std::size_t kSm87BulkV2P40NvFp4DownOuterOperations = 64U;
inline constexpr std::size_t kSm87BulkV2P40NvFp4FusedOuterOperations = 128U;
inline constexpr std::size_t kSm87BulkV2P40NvFp4WholeRoleLaunches = 128U;

inline constexpr std::size_t kSm87BulkV2P40Bf16AbLogicalRoles = 96U;
inline constexpr std::size_t kSm87BulkV2P40Bf16AbFusedOuterOperations = 48U;
inline constexpr std::size_t kSm87BulkV2P40Bf16AbPhysicalLaunches = 48U;

inline constexpr std::size_t kSm87BulkV2P40ProjectionLogicalRoles = 496U;
inline constexpr std::size_t kSm87BulkV2P40ProjectionFusedOuterOperations =
    304U;
inline constexpr std::uint64_t
    kSm87BulkV2P40NvFp4ConventionalOperations =
        1'369'020'825'600'000ULL;
inline constexpr std::uint64_t
    kSm87BulkV2P40Fp8ConventionalOperations = 577'136'230'400'000ULL;
inline constexpr std::uint64_t
    kSm87BulkV2P40Bf16AbConventionalOperations = 1'887'436'800'000ULL;
inline constexpr std::uint64_t
    kSm87BulkV2P40SuccessorProjectionConventionalOperations =
        1'948'044'492'800'000ULL;

inline constexpr std::size_t kSm87BulkV2P40Fp8ExactControlLaunches =
    q3x::kernels::kSm87BulkV2Fp8PhysicalLaunches;
inline constexpr std::size_t kSm87BulkV2P40NvFp4ExactControlLaunches =
    q3x::kernels::kSm87BulkV2NvFp4PhysicalMacroLaunches;

enum class Sm87BulkV2P40ProjectionRoute : std::uint8_t {
  kInvalid = 0U,
  kExactControlSteppingStones,
  kWholeP40000Successors,
};

struct Sm87BulkV2P40ProjectionSuccessorManifest final {
  std::array<std::uint8_t, 8U> magic{};
  std::uint16_t abi_major = 0U;
  std::uint16_t abi_minor = 0U;
  std::size_t prompt_tokens = 0U;
  std::size_t fp8_logical_roles = 0U;
  std::size_t fp8_fused_outer_operations = 0U;
  std::size_t fp8_whole_role_launches = 0U;
  std::size_t nvfp4_logical_roles = 0U;
  std::size_t nvfp4_fused_outer_operations = 0U;
  std::size_t nvfp4_whole_role_launches = 0U;
  std::size_t bf16_ab_logical_roles = 0U;
  std::size_t bf16_ab_fused_outer_operations = 0U;
  std::size_t bf16_ab_physical_launches = 0U;
  std::size_t logical_projection_roles = 0U;
  std::size_t fused_outer_operations = 0U;
  std::uint64_t conventional_operations = 0U;
  std::size_t fp8_exact_control_launches = 0U;
  std::size_t nvfp4_exact_control_launches = 0U;
  bool one_whole_launch_per_quantized_outer_operation = false;
  bool exact_controls_are_oracle_only = false;
  bool exact_controls_can_satisfy_successor_receipt = false;
  bool default_off = false;
  bool numerical_contract_qualified = false;
  bool static_resource_contract_qualified = false;
  bool production_dispatch_eligible = false;
};

[[nodiscard]] constexpr Sm87BulkV2P40ProjectionSuccessorManifest
sm87_bulk_v2_p40_projection_successor_manifest() noexcept {
  return {
      kSm87BulkV2P40ProjectionSuccessorMagic,
      kSm87BulkV2P40ProjectionSuccessorAbiMajor,
      kSm87BulkV2P40ProjectionSuccessorAbiMinor,
      kSm87BulkV2P40ProjectionTokens,
      kSm87BulkV2P40Fp8LogicalRoles,
      kSm87BulkV2P40Fp8FusedOuterOperations,
      kSm87BulkV2P40Fp8WholeRoleLaunches,
      kSm87BulkV2P40NvFp4LogicalRoles,
      kSm87BulkV2P40NvFp4FusedOuterOperations,
      kSm87BulkV2P40NvFp4WholeRoleLaunches,
      kSm87BulkV2P40Bf16AbLogicalRoles,
      kSm87BulkV2P40Bf16AbFusedOuterOperations,
      kSm87BulkV2P40Bf16AbPhysicalLaunches,
      kSm87BulkV2P40ProjectionLogicalRoles,
      kSm87BulkV2P40ProjectionFusedOuterOperations,
      kSm87BulkV2P40SuccessorProjectionConventionalOperations,
      kSm87BulkV2P40Fp8ExactControlLaunches,
      kSm87BulkV2P40NvFp4ExactControlLaunches,
      true,
      true,
      false,
      true,
      false,
      false,
      false,
  };
}

[[nodiscard]] constexpr bool
sm87_bulk_v2_p40_projection_magic_equal(
    const std::array<std::uint8_t, 8U>& left,
    const std::array<std::uint8_t, 8U>& right) noexcept {
  for (std::size_t index = 0U; index < left.size(); ++index) {
    if (left[index] != right[index]) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] constexpr bool
sm87_bulk_v2_p40_projection_successor_manifest_valid(
    const Sm87BulkV2P40ProjectionSuccessorManifest& manifest) noexcept {
  constexpr auto expected =
      sm87_bulk_v2_p40_projection_successor_manifest();
  return sm87_bulk_v2_p40_projection_magic_equal(manifest.magic,
                                                  expected.magic) &&
         manifest.abi_major == expected.abi_major &&
         manifest.abi_minor == expected.abi_minor &&
         manifest.prompt_tokens == expected.prompt_tokens &&
         manifest.fp8_logical_roles == expected.fp8_logical_roles &&
         manifest.fp8_fused_outer_operations ==
             expected.fp8_fused_outer_operations &&
         manifest.fp8_whole_role_launches ==
             expected.fp8_whole_role_launches &&
         manifest.nvfp4_logical_roles == expected.nvfp4_logical_roles &&
         manifest.nvfp4_fused_outer_operations ==
             expected.nvfp4_fused_outer_operations &&
         manifest.nvfp4_whole_role_launches ==
             expected.nvfp4_whole_role_launches &&
         manifest.bf16_ab_logical_roles == expected.bf16_ab_logical_roles &&
         manifest.bf16_ab_fused_outer_operations ==
             expected.bf16_ab_fused_outer_operations &&
         manifest.bf16_ab_physical_launches ==
             expected.bf16_ab_physical_launches &&
         manifest.logical_projection_roles ==
             expected.logical_projection_roles &&
         manifest.fused_outer_operations == expected.fused_outer_operations &&
         manifest.conventional_operations ==
             expected.conventional_operations &&
         manifest.fp8_exact_control_launches ==
             expected.fp8_exact_control_launches &&
         manifest.nvfp4_exact_control_launches ==
             expected.nvfp4_exact_control_launches &&
         manifest.one_whole_launch_per_quantized_outer_operation &&
         manifest.exact_controls_are_oracle_only &&
         !manifest.exact_controls_can_satisfy_successor_receipt &&
         manifest.default_off && !manifest.numerical_contract_qualified &&
         !manifest.static_resource_contract_qualified &&
         !manifest.production_dispatch_eligible;
}

// This caller-fillable value is an observation checked at terminal
// completion, never a launch or startup capability.  Only the private
// owner-issued execution access may authorize CUDA submission; constructing
// this struct grants no access to FP8, Gate+Up, or Down launchers.
struct Sm87BulkV2P40ProjectionSuccessorReceipt final {
  std::array<std::uint8_t, 8U> magic{};
  std::uint16_t abi_major = 0U;
  std::uint16_t abi_minor = 0U;
  Sm87BulkV2P40ProjectionRoute route =
      Sm87BulkV2P40ProjectionRoute::kInvalid;
  std::size_t prompt_tokens = 0U;
  std::size_t fp8_gdn_input_whole_launches = 0U;
  std::size_t fp8_full_input_whole_launches = 0U;
  std::size_t fp8_output_whole_launches = 0U;
  std::size_t fp8_whole_role_launches = 0U;
  std::size_t nvfp4_gate_up_whole_launches = 0U;
  std::size_t nvfp4_down_whole_launches = 0U;
  std::size_t nvfp4_whole_role_launches = 0U;
  std::size_t bf16_ab_physical_launches = 0U;
  std::size_t logical_projection_roles = 0U;
  std::size_t fused_outer_operations = 0U;
  std::uint64_t conventional_operations = 0U;
  std::size_t fp8_exact_control_launches = 0U;
  std::size_t nvfp4_exact_control_launches = 0U;
  bool default_off_route = false;
  bool numerical_contract_qualified = false;
  bool static_resource_contract_qualified = false;
  bool production_dispatch_eligible = false;
};

[[nodiscard]] constexpr Sm87BulkV2P40ProjectionSuccessorReceipt
sm87_bulk_v2_p40_projection_successor_receipt() noexcept {
  Sm87BulkV2P40ProjectionSuccessorReceipt receipt;
  receipt.magic = kSm87BulkV2P40ProjectionSuccessorMagic;
  receipt.abi_major = kSm87BulkV2P40ProjectionSuccessorAbiMajor;
  receipt.abi_minor = kSm87BulkV2P40ProjectionSuccessorAbiMinor;
  receipt.route = Sm87BulkV2P40ProjectionRoute::kWholeP40000Successors;
  receipt.prompt_tokens = kSm87BulkV2P40ProjectionTokens;
  receipt.logical_projection_roles =
      kSm87BulkV2P40ProjectionLogicalRoles;
  receipt.fused_outer_operations =
      kSm87BulkV2P40ProjectionFusedOuterOperations;
  receipt.conventional_operations =
      kSm87BulkV2P40SuccessorProjectionConventionalOperations;
  receipt.default_off_route = true;
  receipt.numerical_contract_qualified = false;
  receipt.static_resource_contract_qualified = false;
  receipt.production_dispatch_eligible = false;
  return receipt;
}

[[nodiscard]] constexpr bool
sm87_bulk_v2_p40_projection_successor_receipt_complete(
    const Sm87BulkV2P40ProjectionSuccessorReceipt& receipt) noexcept {
  return sm87_bulk_v2_p40_projection_magic_equal(
             receipt.magic, kSm87BulkV2P40ProjectionSuccessorMagic) &&
         receipt.abi_major == kSm87BulkV2P40ProjectionSuccessorAbiMajor &&
         receipt.abi_minor == kSm87BulkV2P40ProjectionSuccessorAbiMinor &&
         receipt.route ==
             Sm87BulkV2P40ProjectionRoute::kWholeP40000Successors &&
         receipt.prompt_tokens == kSm87BulkV2P40ProjectionTokens &&
         receipt.fp8_gdn_input_whole_launches ==
             kSm87BulkV2P40Fp8GdnInputOuterOperations &&
         receipt.fp8_full_input_whole_launches ==
             kSm87BulkV2P40Fp8FullInputOuterOperations &&
         receipt.fp8_output_whole_launches ==
             kSm87BulkV2P40Fp8OutputOuterOperations &&
         receipt.fp8_whole_role_launches ==
             kSm87BulkV2P40Fp8WholeRoleLaunches &&
         receipt.nvfp4_gate_up_whole_launches ==
             kSm87BulkV2P40NvFp4GateUpOuterOperations &&
         receipt.nvfp4_down_whole_launches ==
             kSm87BulkV2P40NvFp4DownOuterOperations &&
         receipt.nvfp4_whole_role_launches ==
             kSm87BulkV2P40NvFp4WholeRoleLaunches &&
         receipt.bf16_ab_physical_launches ==
             kSm87BulkV2P40Bf16AbPhysicalLaunches &&
         receipt.logical_projection_roles ==
             kSm87BulkV2P40ProjectionLogicalRoles &&
         receipt.fused_outer_operations ==
             kSm87BulkV2P40ProjectionFusedOuterOperations &&
         receipt.conventional_operations ==
             kSm87BulkV2P40SuccessorProjectionConventionalOperations &&
         receipt.fp8_exact_control_launches == 0U &&
         receipt.nvfp4_exact_control_launches == 0U &&
         receipt.default_off_route &&
         !receipt.numerical_contract_qualified &&
         !receipt.static_resource_contract_qualified &&
         !receipt.production_dispatch_eligible;
}

inline constexpr auto kSm87BulkV2P40FrozenProjectionSuccessorManifest =
    sm87_bulk_v2_p40_projection_successor_manifest();

static_assert(kSm87BulkV2P40Fp8GdnInputLogicalRoles +
                      kSm87BulkV2P40Fp8FullInputLogicalRoles +
                      kSm87BulkV2P40Fp8OutputLogicalRoles ==
                  kSm87BulkV2P40Fp8LogicalRoles &&
              kSm87BulkV2P40Fp8GdnInputOuterOperations +
                      kSm87BulkV2P40Fp8FullInputOuterOperations +
                      kSm87BulkV2P40Fp8OutputOuterOperations ==
                  kSm87BulkV2P40Fp8WholeRoleLaunches);
static_assert(kSm87BulkV2P40NvFp4GateUpLogicalRoles +
                      kSm87BulkV2P40NvFp4DownLogicalRoles ==
                  kSm87BulkV2P40NvFp4LogicalRoles &&
              kSm87BulkV2P40NvFp4GateUpOuterOperations +
                      kSm87BulkV2P40NvFp4DownOuterOperations ==
                  kSm87BulkV2P40NvFp4WholeRoleLaunches);
static_assert(kSm87BulkV2P40Fp8LogicalRoles +
                      kSm87BulkV2P40NvFp4LogicalRoles +
                      kSm87BulkV2P40Bf16AbLogicalRoles ==
                  kSm87BulkV2P40ProjectionLogicalRoles &&
              kSm87BulkV2P40Fp8FusedOuterOperations +
                      kSm87BulkV2P40NvFp4FusedOuterOperations +
                      kSm87BulkV2P40Bf16AbFusedOuterOperations ==
                  kSm87BulkV2P40ProjectionFusedOuterOperations);
static_assert(kSm87BulkV2P40NvFp4ConventionalOperations +
                      kSm87BulkV2P40Fp8ConventionalOperations +
                      kSm87BulkV2P40Bf16AbConventionalOperations ==
                  kSm87BulkV2P40SuccessorProjectionConventionalOperations);
static_assert(kSm87BulkV2P40ProjectionTokens ==
                  q3x::kernels::kSm87BulkV2Fp8WholeP40Tokens &&
              kSm87BulkV2P40ProjectionTokens ==
                  q3x::kernels::kSm87BulkV2NvFp4GateUpWholeP40Tokens &&
              kSm87BulkV2P40ProjectionTokens ==
                  q3x::kernels::kSm87BulkV2NvFp4DownWholeP40Tokens);
static_assert(kSm87BulkV2P40Fp8WholeRoleLaunches ==
              q3x::kernels::kSm87BulkV2Fp8WholeP40PhysicalLaunches);
static_assert(
    q3x::kernels::sm87_bulk_v2_nvfp4_gate_up_whole_p40_traffic()
            .physical_launches == 1U &&
    q3x::kernels::sm87_bulk_v2_nvfp4_down_whole_p40_traffic_contract()
            .physical_launches == 1U);
static_assert(kSm87BulkV2P40Fp8ExactControlLaunches == 5'120U &&
              kSm87BulkV2P40NvFp4ExactControlLaunches == 2'560U);
static_assert(sm87_bulk_v2_p40_projection_successor_manifest_valid(
    kSm87BulkV2P40FrozenProjectionSuccessorManifest));

}  // namespace q3x::runtime
