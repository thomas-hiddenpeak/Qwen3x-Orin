#include "sm87_bulk_dataflow_v2_p40_constituent_seal_internal.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <new>

namespace q3x::runtime::sm87_bulk_v2_p40_owner_detail {

class Sm87BulkV2P40WholeProjectionStartupIssuer final {
 public:
  [[nodiscard]] static bool install(
      Sm87BulkV2P40WholeProjectionStartupResult* result,
      Sm87BulkV2P40Owner* owner,
      Sm87BulkV2P40WholeProjectionStartupExecutionClass execution_class)
      noexcept;
};

namespace {

enum class WholeProjectionObservationProvenance : std::uint8_t {
  kCallerFilled = 0U,
  kFixedAotPrivateQuery,
  kSyntheticHostQuery,
};

using WholeProjectionObservations =
    Sm87BulkV2P40WholeProjectionStartupObservations;

[[nodiscard]] constexpr
Sm87BulkV2P40WholeProjectionRetainedEvidenceCatalog
frozen_whole_projection_retained_evidence_catalog() noexcept {
  return {
      {{'Q', '3', 'X', 'V', '2', 'P', '4', '0'}},
      1U,
      0U,
      // Retained fp8-whole-p40.sass evidence record.  This is not a hash of
      // the currently loaded executable image.
      {{0x02U, 0xdcU, 0x9bU, 0x33U, 0xeaU, 0x42U, 0x50U, 0xacU,
        0x06U, 0x81U, 0xf0U, 0x3bU, 0x94U, 0x6bU, 0x77U, 0xc4U,
        0x79U, 0x43U, 0xc4U, 0xc1U, 0x44U, 0xd3U, 0xd5U, 0x48U,
        0x98U, 0xfbU, 0xb7U, 0xe3U, 0xbeU, 0xfcU, 0xedU, 0xebU}},
      // Retained Gate+Up candidate cubin evidence record.
      {{0xc5U, 0xa1U, 0xc2U, 0x49U, 0xb2U, 0x71U, 0x9cU, 0xc9U,
        0x81U, 0xc6U, 0x00U, 0x79U, 0xe6U, 0x17U, 0xaeU, 0x2cU,
        0xbeU, 0xb4U, 0x7aU, 0xebU, 0xb8U, 0xd6U, 0xdfU, 0xa8U,
        0x4bU, 0x37U, 0x01U, 0x85U, 0xf6U, 0xe3U, 0xa2U, 0xe0U}},
      // Retained canonical Gate+Up cuobjdump SASS evidence record.
      {{0xf6U, 0x79U, 0xf0U, 0xbeU, 0x88U, 0x33U, 0xf5U, 0x43U,
        0xf6U, 0xdeU, 0xbeU, 0xdfU, 0x05U, 0x3eU, 0xfeU, 0x93U,
        0xc9U, 0xaeU, 0x92U, 0x02U, 0x85U, 0xa6U, 0x13U, 0xf9U,
        0x31U, 0xc8U, 0x13U, 0x58U, 0x97U, 0xb0U, 0xb1U, 0xb3U}},
      // Retained canonical Down cuobjdump SASS evidence record.
      {{0x4fU, 0xc1U, 0xf9U, 0x6eU, 0xf2U, 0x70U, 0xd6U, 0xb3U,
        0x03U, 0x12U, 0xc8U, 0xcaU, 0x72U, 0x6cU, 0x74U, 0x99U,
        0x45U, 0x6eU, 0x6cU, 0x8bU, 0xd4U, 0x25U, 0x04U, 0x99U,
        0xdfU, 0xa1U, 0xb5U, 0x07U, 0xb4U, 0x25U, 0x7eU, 0x35U}},
  };
}

[[nodiscard]] constexpr q3x::kernels::Sm87BulkV2Fp8WholeP40CodeEvidence
frozen_fp8_code_evidence() noexcept {
  q3x::kernels::Sm87BulkV2Fp8WholeP40CodeEvidence result;
  // This retained evidence record is compiled into the source-private root;
  // callers cannot supply it.  The linked-symbol query below observes actual
  // CUDA resources but does not hash the loaded ELF/cubin/SASS, so these
  // fields are catalog-consistency inputs rather than binary authentication.
  result.elf_identity = 0x1e98'3c5a'abea'05beULL;
  result.sass_identity = 0x02dc'9b33'ea42'50acULL;
  result.launch_bounds_256_2 = true;
  result.contains_cp_async_cg = true;
  result.contains_ldmatrix = true;
  result.contains_bf16_mma = true;
  result.same_kernel_exact_oracle = true;
  result.no_partial_c_symbol = true;
  result.valid = true;
  return result;
}

[[nodiscard]] constexpr
q3x::kernels::Sm87BulkV2NvFp4GateUpWholeP40CodeEvidence
frozen_gate_up_code_evidence() noexcept {
  q3x::kernels::Sm87BulkV2NvFp4GateUpWholeP40CodeEvidence result;
  result.elf_identity = 0xc5a1'c249'b271'9cc9ULL;
  result.canonical_sass_hash = 0xf679'f0be'8833'f543ULL;
  result.instruction_rows = 5'256U;
  result.text_bytes = 84'096U;
  result.contains_cp_async_cg = true;
  result.contains_ldmatrix = true;
  result.contains_bf16_mma = true;
  result.one_cooperative_kernel_symbol = true;
  result.same_elf_exact_oracle = true;
  result.valid = true;
  return result;
}

[[nodiscard]] constexpr
q3x::kernels::Sm87BulkV2NvFp4DownWholeP40CodeEvidence
frozen_down_code_evidence() noexcept {
  q3x::kernels::Sm87BulkV2NvFp4DownWholeP40CodeEvidence result;
  result.elf_identity = 0x22fc'363b'192d'8cf8ULL;
  result.canonical_sass_hash = 0x4fc1'f96e'f270'd6b3ULL;
  result.instruction_rows = 7'904U;
  result.text_bytes = 126'464U;
  result.launch_bounds_256_2 = true;
  result.cooperative_grid_sync_present = true;
  result.cp_async_cg_present = true;
  result.two_stage_s2r_present = true;
  result.full_k_accumulator_present = true;
  result.split_k_or_partial_c_absent = true;
  return result;
}

template <std::size_t Size>
[[nodiscard]] bool bytes_nonzero(
    const std::array<std::uint8_t, Size>& bytes) noexcept {
  return std::any_of(bytes.begin(), bytes.end(),
                     [](const std::uint8_t byte) noexcept {
                       return byte != 0U;
                     });
}

[[nodiscard]] constexpr bool same_fp8_code(
    const q3x::kernels::Sm87BulkV2Fp8WholeP40CodeEvidence& left,
    const q3x::kernels::Sm87BulkV2Fp8WholeP40CodeEvidence& right) noexcept {
  return left.elf_identity == right.elf_identity &&
         left.sass_identity == right.sass_identity &&
         left.stack_bytes == right.stack_bytes &&
         left.spill_store_bytes == right.spill_store_bytes &&
         left.spill_load_bytes == right.spill_load_bytes &&
         left.local_bytes == right.local_bytes &&
         left.launch_bounds_256_2 == right.launch_bounds_256_2 &&
         left.contains_cp_async_cg == right.contains_cp_async_cg &&
         left.contains_ldmatrix == right.contains_ldmatrix &&
         left.contains_bf16_mma == right.contains_bf16_mma &&
         left.same_kernel_exact_oracle == right.same_kernel_exact_oracle &&
         left.no_partial_c_symbol == right.no_partial_c_symbol &&
         left.valid == right.valid;
}

[[nodiscard]] constexpr bool same_gate_up_code(
    const q3x::kernels::Sm87BulkV2NvFp4GateUpWholeP40CodeEvidence& left,
    const q3x::kernels::Sm87BulkV2NvFp4GateUpWholeP40CodeEvidence& right)
    noexcept {
  return left.elf_identity == right.elf_identity &&
         left.canonical_sass_hash == right.canonical_sass_hash &&
         left.instruction_rows == right.instruction_rows &&
         left.text_bytes == right.text_bytes &&
         left.stack_bytes == right.stack_bytes &&
         left.spill_store_bytes == right.spill_store_bytes &&
         left.spill_load_bytes == right.spill_load_bytes &&
         left.local_bytes == right.local_bytes &&
         left.contains_cp_async_cg == right.contains_cp_async_cg &&
         left.contains_ldmatrix == right.contains_ldmatrix &&
         left.contains_bf16_mma == right.contains_bf16_mma &&
         left.one_cooperative_kernel_symbol ==
             right.one_cooperative_kernel_symbol &&
         left.same_elf_exact_oracle == right.same_elf_exact_oracle &&
         left.valid == right.valid;
}

[[nodiscard]] constexpr bool same_down_code(
    const q3x::kernels::Sm87BulkV2NvFp4DownWholeP40CodeEvidence& left,
    const q3x::kernels::Sm87BulkV2NvFp4DownWholeP40CodeEvidence& right)
    noexcept {
  return left.elf_identity == right.elf_identity &&
         left.canonical_sass_hash == right.canonical_sass_hash &&
         left.instruction_rows == right.instruction_rows &&
         left.text_bytes == right.text_bytes &&
         left.stack_frame_bytes == right.stack_frame_bytes &&
         left.spill_store_bytes == right.spill_store_bytes &&
         left.spill_load_bytes == right.spill_load_bytes &&
         left.local_load_store_rows == right.local_load_store_rows &&
         left.launch_bounds_256_2 == right.launch_bounds_256_2 &&
         left.cooperative_grid_sync_present ==
             right.cooperative_grid_sync_present &&
         left.cp_async_cg_present == right.cp_async_cg_present &&
         left.two_stage_s2r_present == right.two_stage_s2r_present &&
         left.full_k_accumulator_present ==
             right.full_k_accumulator_present &&
         left.split_k_or_partial_c_absent ==
             right.split_k_or_partial_c_absent;
}

[[nodiscard]] constexpr bool fp8_runtime_resources_exact(
    const q3x::kernels::Sm87BulkV2Fp8WholeP40FamilyResources& resources)
    noexcept {
  constexpr std::array<q3x::kernels::Sm87TargetAotProjectionRole, 3U> roles{{
      q3x::kernels::Sm87TargetAotProjectionRole::kFp8GdnQkvZ,
      q3x::kernels::Sm87TargetAotProjectionRole::kFp8FullQkv,
      q3x::kernels::Sm87TargetAotProjectionRole::kFp8AttentionOutput,
  }};
  constexpr std::array<int, 3U> registers{{90, 93, 89}};
  for (std::size_t index = 0U; index < resources.roles.size(); ++index) {
    const auto& role = resources.roles[index];
    if (role.role != roles[index] || role.binary_version != 87 ||
        role.registers_per_thread != registers[index] ||
        role.static_shared_bytes != 0U ||
        role.dynamic_shared_bytes !=
            q3x::kernels::kSm87BulkV2Fp8WholeP40DynamicSharedBytes ||
        role.local_bytes != 0U || role.maximum_threads_per_block != 256 ||
        role.active_blocks_per_sm != 2 ||
        role.cooperative_grid_capacity != 32 ||
        !role.cooperative_launch_supported ||
        !role.runtime_envelope_observed ||
        !role.external_static_record_consistent ||
        role.admission_capability_issued ||
        role.numerical_contract_qualified ||
        role.production_dispatch_eligible) {
      return false;
    }
  }
  return resources.all_runtime_envelopes_observed &&
         resources.all_external_static_records_consistent &&
         !resources.admission_capability_issued &&
         !resources.numerical_contract_qualified &&
         !resources.production_dispatch_eligible;
}

[[nodiscard]] constexpr bool gate_up_runtime_resources_exact(
    const q3x::kernels::Sm87BulkV2NvFp4GateUpWholeP40Resources& resources)
    noexcept {
  return resources.binary_version == 87 &&
         resources.registers_per_thread == 107 &&
         resources.static_shared_bytes == 0U &&
         resources.dynamic_shared_bytes ==
             q3x::kernels::
                 kSm87BulkV2NvFp4GateUpWholeP40DynamicSharedBytes &&
         resources.local_bytes == 0U &&
         resources.maximum_threads_per_block == 256 &&
         resources.active_blocks_per_sm == 2 &&
         resources.cooperative_grid_capacity == 32 &&
         resources.cooperative_launch_supported &&
         resources.exact_oracle_attached && !resources.resource_gate_passed &&
         !resources.numerical_contract_qualified &&
         !resources.production_dispatch_eligible;
}

[[nodiscard]] constexpr bool down_runtime_resources_exact(
    const q3x::kernels::Sm87BulkV2NvFp4DownWholeP40Resources& resources,
    const std::int32_t device_ordinal) noexcept {
  return resources.kernel_symbol_identity ==
             q3x::kernels::
                 kSm87BulkV2NvFp4DownWholeP40KernelSymbolIdentity &&
         resources.device_ordinal == device_ordinal &&
         resources.binary_version == 87 &&
         resources.registers_per_thread == 111 &&
         resources.static_shared_bytes == 0U &&
         resources.dynamic_shared_bytes ==
             q3x::kernels::kSm87BulkV2NvFp4DownWholeP40DynamicSharedBytes &&
         resources.local_bytes == 0U &&
         resources.maximum_threads_per_block == 256 &&
         resources.active_blocks_per_sm == 2 &&
         resources.cooperative_grid_capacity == 32 &&
         resources.kernel_compiled &&
         resources.cooperative_launch_supported &&
         resources.dynamic_shared_attribute_configured &&
         !resources.resource_gate_passed &&
         !resources.numerical_contract_qualified &&
         !resources.production_dispatch_eligible;
}

[[nodiscard]] bool retained_evidence_records_match_catalog(
    const WholeProjectionObservations& observations) noexcept {
  constexpr auto fp8 = frozen_fp8_code_evidence();
  constexpr auto gate_up = frozen_gate_up_code_evidence();
  constexpr auto down = frozen_down_code_evidence();
  return same_fp8_code(observations.fp8.roles[0U].code, fp8) &&
         same_fp8_code(observations.fp8.roles[1U].code, fp8) &&
         same_fp8_code(observations.fp8.roles[2U].code, fp8) &&
         same_gate_up_code(observations.gate_up.code, gate_up) &&
         same_down_code(observations.down.code, down);
}

#if defined(Q3X_ENABLE_SM87_BULK_DATAFLOW_V2_P40_STARTUP_SEAL_HOST_FIXTURE)
[[nodiscard]] WholeProjectionObservations passing_host_observations(
    const std::int32_t device_ordinal) noexcept {
  WholeProjectionObservations observations;
  constexpr std::array<q3x::kernels::Sm87TargetAotProjectionRole, 3U> roles{{
      q3x::kernels::Sm87TargetAotProjectionRole::kFp8GdnQkvZ,
      q3x::kernels::Sm87TargetAotProjectionRole::kFp8FullQkv,
      q3x::kernels::Sm87TargetAotProjectionRole::kFp8AttentionOutput,
  }};
  constexpr std::array<int, 3U> registers{{90, 93, 89}};
  constexpr auto fp8_code = frozen_fp8_code_evidence();
  for (std::size_t index = 0U; index < observations.fp8.roles.size();
       ++index) {
    auto& resource = observations.fp8.roles[index];
    resource.role = roles[index];
    resource.binary_version = 87;
    resource.registers_per_thread = registers[index];
    resource.dynamic_shared_bytes =
        q3x::kernels::kSm87BulkV2Fp8WholeP40DynamicSharedBytes;
    resource.maximum_threads_per_block = 256;
    resource.active_blocks_per_sm = 2;
    resource.cooperative_grid_capacity = 32;
    resource.code = fp8_code;
    resource.cooperative_launch_supported = true;
    resource.runtime_envelope_observed = true;
    resource.external_static_record_consistent = true;
    resource.admission_capability_issued = false;
    resource.numerical_contract_qualified = false;
    resource.production_dispatch_eligible = false;
  }
  observations.fp8.all_runtime_envelopes_observed = true;
  observations.fp8.all_external_static_records_consistent = true;
  observations.fp8.admission_capability_issued = false;
  observations.fp8.numerical_contract_qualified = false;
  observations.fp8.production_dispatch_eligible = false;

  observations.gate_up.binary_version = 87;
  observations.gate_up.registers_per_thread = 107;
  observations.gate_up.dynamic_shared_bytes =
      q3x::kernels::kSm87BulkV2NvFp4GateUpWholeP40DynamicSharedBytes;
  observations.gate_up.maximum_threads_per_block = 256;
  observations.gate_up.active_blocks_per_sm = 2;
  observations.gate_up.cooperative_grid_capacity = 32;
  observations.gate_up.code = frozen_gate_up_code_evidence();
  observations.gate_up.cooperative_launch_supported = true;
  observations.gate_up.exact_oracle_attached = true;
  observations.gate_up.resource_gate_passed = false;
  observations.gate_up.numerical_contract_qualified = false;
  observations.gate_up.production_dispatch_eligible = false;

  observations.down.kernel_symbol_identity =
      q3x::kernels::kSm87BulkV2NvFp4DownWholeP40KernelSymbolIdentity;
  observations.down.device_ordinal = device_ordinal;
  observations.down.binary_version = 87;
  observations.down.registers_per_thread = 111;
  observations.down.dynamic_shared_bytes =
      q3x::kernels::kSm87BulkV2NvFp4DownWholeP40DynamicSharedBytes;
  observations.down.maximum_threads_per_block = 256;
  observations.down.active_blocks_per_sm = 2;
  observations.down.cooperative_grid_capacity = 32;
  observations.down.code = frozen_down_code_evidence();
  observations.down.kernel_compiled = true;
  observations.down.cooperative_launch_supported = true;
  observations.down.dynamic_shared_attribute_configured = true;
  observations.down.resource_gate_passed = false;
  observations.down.numerical_contract_qualified = false;
  observations.down.production_dispatch_eligible = false;

  observations.fp8_successor_linked = true;
  observations.gate_up_successor_linked = true;
  observations.down_successor_linked = true;
  return observations;
}
#endif

[[nodiscard, maybe_unused]] Sm87BulkV2P40WholeProjectionStartupResult
issue_whole_projection_startup_root(
    Sm87BulkV2P40Owner& owner,
    const WholeProjectionObservations& observations,
    const WholeProjectionObservationProvenance provenance) noexcept {
  Sm87BulkV2P40WholeProjectionStartupResult result;
  result.audit.retained_catalog =
      frozen_whole_projection_retained_evidence_catalog();
  result.audit.owner_identity = owner.owner_identity();
  result.audit.device_ordinal = owner.device_ordinal();
  result.audit.fp8_whole_successor_linked =
      observations.fp8_successor_linked;
  result.audit.gate_up_whole_successor_linked =
      observations.gate_up_successor_linked;
  result.audit.down_whole_successor_linked =
      observations.down_successor_linked;
  result.audit.default_off = true;
  result.audit.numerical_contract_qualified = false;
  result.audit.performance_qualified = false;
  result.audit.release_qualified = false;
  result.audit.production_dispatch_eligible = false;

  if (owner.state() != Sm87BulkV2P40OwnerState::kResourcesReady ||
      owner.execution_access() != nullptr || owner.owner_identity() == 0U ||
      owner.device_ordinal() < 0) {
    result.status = {
        Sm87BulkV2P40WholeProjectionStartupError::kInvalidOwner,
        "whole_projection_startup_requires_unsealed_resource_owner", 0,
        0U};
    return result;
  }
  if (provenance == WholeProjectionObservationProvenance::kCallerFilled) {
    result.status = {
        Sm87BulkV2P40WholeProjectionStartupError::
            kCallerFilledObservationIsNotAuthority,
        "caller_filled_whole_projection_resources_are_observation_only", 0,
        0U};
    return result;
  }
  result.audit.private_startup_query_used = true;
  result.audit.caller_filled_public_observation_used_as_authority = false;
  result.audit.cuda_resource_queries = 3U;
  result.audit.configured_source_sha256_gate_passed =
      provenance == WholeProjectionObservationProvenance::kFixedAotPrivateQuery;
  result.audit.synthetic_host_query =
      provenance == WholeProjectionObservationProvenance::kSyntheticHostQuery;

  if (!observations.fp8_successor_linked ||
      !observations.gate_up_successor_linked ||
      !observations.down_successor_linked) {
    const std::size_t missing = !observations.fp8_successor_linked
                                    ? 0U
                                    : (!observations.gate_up_successor_linked
                                           ? 3U
                                           : 4U);
    result.status = {
        Sm87BulkV2P40WholeProjectionStartupError::kMissingWholeSuccessor,
        "complete_whole_projection_successor_set_required", 0, missing};
    return result;
  }
  if (observations.fp8_query_error != 0) {
    result.status = {Sm87BulkV2P40WholeProjectionStartupError::kFp8Query,
                     "whole_fp8_startup_resource_query",
                     observations.fp8_query_error, 0U};
    return result;
  }
  if (observations.gate_up_query_error != 0) {
    result.status = {
        Sm87BulkV2P40WholeProjectionStartupError::kGateUpQuery,
        "whole_gate_up_startup_resource_query",
        observations.gate_up_query_error, 3U};
    return result;
  }
  if (observations.down_query_error != 0) {
    result.status = {Sm87BulkV2P40WholeProjectionStartupError::kDownQuery,
                     "whole_down_startup_resource_query",
                     observations.down_query_error, 4U};
    return result;
  }
  if (observations.down.device_ordinal != owner.device_ordinal()) {
    result.status = {Sm87BulkV2P40WholeProjectionStartupError::kWrongDevice,
                     "whole_projection_query_device_mismatch", 0, 4U};
    return result;
  }

  const bool fp8_resources = fp8_runtime_resources_exact(observations.fp8);
  const bool gate_up_resources =
      gate_up_runtime_resources_exact(observations.gate_up);
  const bool down_resources = down_runtime_resources_exact(
      observations.down, owner.device_ordinal());
  if (!fp8_resources || !gate_up_resources || !down_resources) {
    const std::size_t failed = !fp8_resources ? 0U :
                               (!gate_up_resources ? 3U : 4U);
    result.status = {
        Sm87BulkV2P40WholeProjectionStartupError::kResourceMismatch,
        "whole_projection_exact_runtime_resource_mismatch", 0, failed};
    return result;
  }
  result.audit.all_required_dynamic_shared_attributes_configured = true;
  result.audit.all_linked_symbol_resource_envelopes_exact = true;

  if (!result.audit.retained_catalog.valid() ||
      !retained_evidence_records_match_catalog(observations)) {
    result.status = {
        Sm87BulkV2P40WholeProjectionStartupError::
            kRetainedEvidenceCatalogMismatch,
        "whole_projection_retained_evidence_catalog_mismatch", 0, 0U};
    return result;
  }
  result.audit.retained_evidence_catalog_consistent = true;
  if (!result.audit.resource_qualification_valid()) {
    result.status = {
        Sm87BulkV2P40WholeProjectionStartupError::
            kRetainedEvidenceCatalogMismatch,
        "whole_projection_source_resource_envelope_incomplete", 0, 0U};
    return result;
  }

  const auto execution_class =
      provenance == WholeProjectionObservationProvenance::kFixedAotPrivateQuery
          ? Sm87BulkV2P40WholeProjectionStartupExecutionClass::
                kDefaultOffFixedAot
          : Sm87BulkV2P40WholeProjectionStartupExecutionClass::
                kSyntheticHostQuery;
  if (!Sm87BulkV2P40WholeProjectionStartupIssuer::install(
          &result, &owner, execution_class)) {
    result.status = {Sm87BulkV2P40WholeProjectionStartupError::kAllocation,
                     "whole_projection_startup_root_or_access_allocation", 0,
                     0U};
    return result;
  }
  result.status = {};
  return result;
}

[[nodiscard]] constexpr std::size_t constituent_index(
    const Sm87BulkV2P40Constituent constituent) noexcept {
  return static_cast<std::size_t>(constituent);
}

[[nodiscard]] bool all_ready(
    const std::array<Sm87BulkV2P40ConstituentFinding,
                     kSm87BulkV2P40ConstituentCount>& findings) noexcept {
  return std::all_of(
      findings.begin(), findings.end(), [](const auto finding) noexcept {
        return finding == Sm87BulkV2P40ConstituentFinding::kReady;
      });
}

[[nodiscard]] Sm87BulkV2P40ConstituentFinding audit_fp8(
    const std::int32_t owner_device, void* const expected_stream,
    const Sm87BulkV2P40RealConstituentSealRequest& request) noexcept {
  if (request.fp8 == nullptr) {
    return Sm87BulkV2P40ConstituentFinding::kMissingSealedAccess;
  }
  if (!request.fp8->valid()) {
    return Sm87BulkV2P40ConstituentFinding::kInvalidSealedAccess;
  }
  const auto& receipt = request.fp8->receipt();
  if (receipt.deployment_identity != request.identities.deployment_identity) {
    return Sm87BulkV2P40ConstituentFinding::kForeignDeployment;
  }
  if (receipt.device_ordinal != owner_device) {
    return Sm87BulkV2P40ConstituentFinding::kForeignDevice;
  }
  if (receipt.cuda_stream_owner_identity !=
      request.identities.stream_event_owner_identity) {
    return Sm87BulkV2P40ConstituentFinding::kForeignStreamOwner;
  }
  if (receipt.cuda_stream != expected_stream) {
    return Sm87BulkV2P40ConstituentFinding::kWrongOwnedStream;
  }
  if (!receipt.numerical_contract_qualified) {
    return Sm87BulkV2P40ConstituentFinding::kNumericalContractUnqualified;
  }
  if (receipt.production_dispatch_eligible) {
    return Sm87BulkV2P40ConstituentFinding::
        kProductionDispatchMustRemainFalse;
  }
  // The current FP8 receipt carries no identity binding the numerical oracle
  // to this exact binary/access.  A true flag alone is not sufficient.
  return Sm87BulkV2P40ConstituentFinding::
      kMissingAuthenticatedNumericalEvidenceBinding;
}

[[nodiscard]] Sm87BulkV2P40ConstituentFinding audit_attention(
    const std::int32_t owner_device, void* const expected_stream,
    const Sm87BulkV2P40RealConstituentSealRequest& request) noexcept {
  if (request.attention == nullptr) {
    return Sm87BulkV2P40ConstituentFinding::kMissingSealedAccess;
  }
  if (!request.attention->valid()) {
    return Sm87BulkV2P40ConstituentFinding::kInvalidSealedAccess;
  }
  const auto& receipt = request.attention->receipt();
  if (receipt.deployment_identity != request.identities.deployment_identity) {
    return Sm87BulkV2P40ConstituentFinding::kForeignDeployment;
  }
  if (receipt.device_ordinal != owner_device) {
    return Sm87BulkV2P40ConstituentFinding::kForeignDevice;
  }
  if (receipt.cuda_stream_owner_identity !=
      request.identities.stream_event_owner_identity) {
    return Sm87BulkV2P40ConstituentFinding::kForeignStreamOwner;
  }
  if (receipt.cuda_stream != expected_stream) {
    return Sm87BulkV2P40ConstituentFinding::kWrongOwnedStream;
  }
  if (!receipt.numerical_contract_qualified) {
    return Sm87BulkV2P40ConstituentFinding::kNumericalContractUnqualified;
  }
  if (receipt.production_dispatch_eligible) {
    return Sm87BulkV2P40ConstituentFinding::
        kProductionDispatchMustRemainFalse;
  }
  return Sm87BulkV2P40ConstituentFinding::
      kMissingAuthenticatedNumericalEvidenceBinding;
}

[[nodiscard]] Sm87BulkV2P40ConstituentFinding audit_request_state(
    const std::uint64_t owner_identity, const std::int32_t owner_device,
    void* const owner_main_stream,
    const Sm87BulkV2P40RealConstituentSealRequest& request) noexcept {
  if (request.request_state == nullptr) {
    return Sm87BulkV2P40ConstituentFinding::
        kMissingUnforgeableQualificationInterface;
  }
  if (!request.request_state->default_off_development_resource_valid()) {
    return Sm87BulkV2P40ConstituentFinding::kInvalidSealedAccess;
  }
  const auto& identity = request.request_state->identity();
  if (identity.owner_identity != owner_identity) {
    return Sm87BulkV2P40ConstituentFinding::kForeignOwner;
  }
  if (identity.device_ordinal != owner_device) {
    return Sm87BulkV2P40ConstituentFinding::kForeignDevice;
  }
  if (identity.stream_event_owner_identity !=
      request.identities.stream_event_owner_identity) {
    return Sm87BulkV2P40ConstituentFinding::kForeignStreamOwner;
  }
  if (request.request_state->cuda_stream(Sm87BulkV2P40Stream::kMain) !=
      owner_main_stream) {
    return Sm87BulkV2P40ConstituentFinding::kWrongOwnedStream;
  }
  if (identity.allocation_identity !=
      request.identities.request_allocation_identity) {
    return Sm87BulkV2P40ConstituentFinding::kForeignAllocation;
  }
  return Sm87BulkV2P40ConstituentFinding::kReady;
}

}  // namespace

bool Sm87BulkV2P40WholeProjectionRetainedEvidenceCatalog::valid()
    const noexcept {
  constexpr std::array<std::uint8_t, 8U> expected_magic{
      {'Q', '3', 'X', 'V', '2', 'P', '4', '0'}};
  return magic == expected_magic && abi_major == 1U && abi_minor == 0U &&
         bytes_nonzero(fp8_retained_sass_record_sha256) &&
         bytes_nonzero(gate_up_retained_cubin_record_sha256) &&
         bytes_nonzero(gate_up_retained_sass_record_sha256) &&
         bytes_nonzero(down_retained_sass_record_sha256);
}

bool Sm87BulkV2P40WholeProjectionStartupAudit::
    resource_qualification_valid() const noexcept {
  const bool provenance_valid =
      configured_source_sha256_gate_passed != synthetic_host_query;
  return retained_catalog.valid() && owner_identity != 0U &&
         device_ordinal >= 0 &&
         cuda_resource_queries == 3U && fp8_whole_successor_linked &&
         gate_up_whole_successor_linked && down_whole_successor_linked &&
         all_required_dynamic_shared_attributes_configured &&
         all_linked_symbol_resource_envelopes_exact &&
         retained_evidence_catalog_consistent && provenance_valid &&
         private_startup_query_used &&
         !caller_filled_public_observation_used_as_authority && default_off &&
         !numerical_contract_qualified && !performance_qualified &&
         !release_qualified && !production_dispatch_eligible;
}

Sm87BulkV2P40WholeProjectionStartupAccess::
    Sm87BulkV2P40WholeProjectionStartupAccess(
        const Sm87BulkV2P40WholeProjectionStartupRoot* const issuer,
        const Sm87BulkV2P40Owner* const owner,
        const Sm87BulkV2P40WholeProjectionRetainedEvidenceCatalog&
            retained_catalog,
        const Sm87BulkV2P40WholeProjectionStartupExecutionClass
            execution_class,
        const std::uint64_t owner_identity,
        const std::int32_t device_ordinal) noexcept
    : issuer_(issuer),
      owner_(owner),
      retained_catalog_(retained_catalog),
      execution_class_(execution_class),
      owner_identity_(owner_identity),
      device_ordinal_(device_ordinal) {}

bool Sm87BulkV2P40WholeProjectionStartupAccess::
    default_off_fixed_aot_resource_valid() const noexcept {
  return issuer_ != nullptr && issuer_->access() == this && owner_ != nullptr &&
         execution_class_ ==
             Sm87BulkV2P40WholeProjectionStartupExecutionClass::
                 kDefaultOffFixedAot &&
         retained_catalog_.valid() && owner_identity_ != 0U &&
         device_ordinal_ >= 0 &&
         whole_fp8_bound_ && whole_gate_up_bound_ && whole_down_bound_ &&
         required_dynamic_shared_attributes_configured_ &&
         linked_symbol_resource_envelopes_exact_ &&
         retained_evidence_catalog_consistent_ &&
         configured_source_sha256_gate_passed_ && default_off_ &&
         bound_to(*owner_);
}

bool Sm87BulkV2P40WholeProjectionStartupAccess::bound_to(
    const Sm87BulkV2P40Owner& owner) const noexcept {
  return owner_ == &owner && owner_identity_ == owner.owner_identity() &&
         device_ordinal_ == owner.device_ordinal();
}

Sm87BulkV2P40WholeProjectionStartupRoot::
    Sm87BulkV2P40WholeProjectionStartupRoot(
        const Sm87BulkV2P40Owner* const owner) noexcept
    : owner_(owner) {}

bool Sm87BulkV2P40WholeProjectionStartupRoot::install_access(
    const Sm87BulkV2P40WholeProjectionRetainedEvidenceCatalog&
        retained_catalog,
    const Sm87BulkV2P40WholeProjectionStartupExecutionClass execution_class,
    const std::uint64_t owner_identity,
    const std::int32_t device_ordinal) noexcept {
  if (owner_ == nullptr || access_ != nullptr || !retained_catalog.valid() ||
      owner_identity == 0U || device_ordinal < 0) {
    return false;
  }
  auto* const access = new (std::nothrow)
      Sm87BulkV2P40WholeProjectionStartupAccess(
          this, owner_, retained_catalog, execution_class, owner_identity,
          device_ordinal);
  if (access == nullptr) {
    return false;
  }
  access_.reset(access);
  access->whole_fp8_bound_ = true;
  access->whole_gate_up_bound_ = true;
  access->whole_down_bound_ = true;
  access->required_dynamic_shared_attributes_configured_ = true;
  access->linked_symbol_resource_envelopes_exact_ = true;
  access->retained_evidence_catalog_consistent_ = true;
  access->configured_source_sha256_gate_passed_ =
      execution_class == Sm87BulkV2P40WholeProjectionStartupExecutionClass::
                             kDefaultOffFixedAot;
  access->default_off_ = true;
  return true;
}

bool Sm87BulkV2P40WholeProjectionStartupIssuer::install(
    Sm87BulkV2P40WholeProjectionStartupResult* const result,
    Sm87BulkV2P40Owner* const owner,
    const Sm87BulkV2P40WholeProjectionStartupExecutionClass execution_class)
    noexcept {
  if (result == nullptr || owner == nullptr || result->root != nullptr ||
      !result->audit.resource_qualification_valid()) {
    return false;
  }
  auto* const root = new (std::nothrow)
      Sm87BulkV2P40WholeProjectionStartupRoot(owner);
  if (root == nullptr) {
    return false;
  }
  result->root.reset(root);
  if (!root->install_access(result->audit.retained_catalog, execution_class,
                            owner->owner_identity(),
                            owner->device_ordinal())) {
    result->root.reset();
    return false;
  }
  return true;
}

Sm87BulkV2P40WholeProjectionStartupResult
create_sm87_bulk_dataflow_v2_p40_whole_projection_startup_root(
    Sm87BulkV2P40Owner& owner) noexcept {
#if defined(Q3X_ENABLE_SM87_BULK_DATAFLOW_V2_P40_WHOLE_PROJECTION_STARTUP)
  WholeProjectionObservations observations;
  observations.fp8_successor_linked = true;
  observations.gate_up_successor_linked = true;
  observations.down_successor_linked = true;
  constexpr auto fp8_code = frozen_fp8_code_evidence();
  const std::array<q3x::kernels::Sm87BulkV2Fp8WholeP40CodeEvidence, 3U>
      fp8_codes{{fp8_code, fp8_code, fp8_code}};
  constexpr auto gate_up_code = frozen_gate_up_code_evidence();
  constexpr auto down_code = frozen_down_code_evidence();
  observations.fp8_query_error =
      q3x::kernels::
          query_sm87_bulk_dataflow_v2_fp8_whole_p40_resources_cuda(
              &fp8_codes, &observations.fp8);
  if (observations.fp8_query_error == 0) {
    observations.gate_up_query_error =
        q3x::kernels::
            query_sm87_bulk_dataflow_v2_nvfp4_gate_up_whole_p40_resources_cuda(
                &gate_up_code, &observations.gate_up);
  }
  if (observations.fp8_query_error == 0 &&
      observations.gate_up_query_error == 0) {
    observations.down_query_error =
        q3x::kernels::
            query_sm87_bulk_dataflow_v2_nvfp4_down_whole_p40_resources_cuda(
                &down_code, &observations.down);
  }
  return issue_whole_projection_startup_root(
      owner, observations,
      WholeProjectionObservationProvenance::kFixedAotPrivateQuery);
#else
  Sm87BulkV2P40WholeProjectionStartupResult result;
  result.audit.retained_catalog =
      frozen_whole_projection_retained_evidence_catalog();
  result.audit.owner_identity = owner.owner_identity();
  result.audit.device_ordinal = owner.device_ordinal();
  result.audit.default_off = true;
  result.audit.numerical_contract_qualified = false;
  result.audit.performance_qualified = false;
  result.audit.release_qualified = false;
  result.audit.production_dispatch_eligible = false;
  result.status = {
      Sm87BulkV2P40WholeProjectionStartupError::kMissingWholeSuccessor,
      "whole_projection_successor_set_not_linked", 0, 0U};
  return result;
#endif
}

#if defined(Q3X_ENABLE_SM87_BULK_DATAFLOW_V2_P40_STARTUP_SEAL_HOST_FIXTURE)
Sm87BulkV2P40WholeProjectionStartupObservations
Sm87BulkV2P40WholeProjectionStartupHostFixture::passing_observations(
    const std::int32_t device_ordinal) noexcept {
  return passing_host_observations(device_ordinal);
}

Sm87BulkV2P40WholeProjectionStartupResult
Sm87BulkV2P40WholeProjectionStartupHostFixture::
    attempt_from_caller_filled_observations(
        Sm87BulkV2P40Owner& owner,
        const Sm87BulkV2P40WholeProjectionStartupObservations& observations)
        noexcept {
  return issue_whole_projection_startup_root(
      owner, observations,
      WholeProjectionObservationProvenance::kCallerFilled);
}

Sm87BulkV2P40WholeProjectionStartupResult
Sm87BulkV2P40WholeProjectionStartupHostFixture::
    mint_from_synthetic_startup_query(
        Sm87BulkV2P40Owner& owner,
        const Sm87BulkV2P40WholeProjectionStartupObservations& observations)
        noexcept {
  return issue_whole_projection_startup_root(
      owner, observations,
      WholeProjectionObservationProvenance::kSyntheticHostQuery);
}

bool Sm87BulkV2P40WholeProjectionStartupHostFixture::synthetic_access_valid(
    const Sm87BulkV2P40WholeProjectionStartupResult& result,
    const Sm87BulkV2P40Owner& owner) noexcept {
  if (result.root == nullptr || result.root->access() == nullptr ||
      !result.audit.resource_qualification_valid() || !result.status) {
    return false;
  }
  const auto* const access = result.root->access();
  return access->issuer_ == result.root.get() && access->bound_to(owner) &&
         access->execution_class_ ==
             Sm87BulkV2P40WholeProjectionStartupExecutionClass::
                 kSyntheticHostQuery &&
         access->whole_fp8_bound_ && access->whole_gate_up_bound_ &&
         access->whole_down_bound_ &&
         access->required_dynamic_shared_attributes_configured_ &&
         access->linked_symbol_resource_envelopes_exact_ &&
         access->retained_evidence_catalog_consistent_ &&
         !access->configured_source_sha256_gate_passed_ &&
         access->default_off_ &&
         !access->default_off_fixed_aot_resource_valid() &&
         !access->numerical_contract_qualified() &&
         !access->performance_qualified() &&
         !access->production_dispatch_eligible();
}
#endif

bool Sm87BulkV2P40DevelopmentAdmissionEvidence::semantics_valid()
    const noexcept {
  return identity.valid() && identity.development_candidate_valid() &&
         all_ready(constituent_findings) &&
         all_static_resource_checks_complete &&
         authenticated_real_constituents &&
         exact_numerical_contract_qualified &&
         default_off_candidate_eligible &&
         !production_dispatch_eligible && !production_selector_bound &&
         !synthetic_host_contract && !mtp_used &&
         !cublaslt_production_path_used && !request_time_jit_used &&
         !request_time_repack_used && !accuracy_relaxation_used;
}

bool Sm87BulkV2P40RealConstituentIdentityClaims::complete() const noexcept {
  return deployment_identity != 0U && model_identity != 0U &&
         request_allocation_identity != 0U &&
         stream_event_owner_identity != 0U &&
         asset_catalog_identity != 0U && binary_evidence_identity != 0U &&
         fp8_oracle_evidence_identity != 0U &&
         attention_oracle_evidence_identity != 0U &&
         gdn_oracle_evidence_identity != 0U &&
         nvfp4_oracle_evidence_identity != 0U;
}

Sm87BulkV2P40RealConstituentSealResult
seal_sm87_bulk_dataflow_v2_p40_real_constituents(
    Sm87BulkV2P40Owner& owner,
    const Sm87BulkV2P40RealConstituentSealRequest& request) noexcept {
  Sm87BulkV2P40RealConstituentSealResult result;
  result.audit.constituent_findings.fill(
      Sm87BulkV2P40ConstituentFinding::kMissingSealedAccess);

  if (owner.state_ != Sm87BulkV2P40OwnerState::kResourcesReady ||
      owner.execution_access_ != nullptr) {
    result.status = {Sm87BulkV2P40OwnerError::kInvalidOwnerState,
                     "real_composite_requires_resources_ready", 0, 0U};
    return result;
  }
  if (!request.identities.complete()) {
    result.status = {Sm87BulkV2P40OwnerError::kMissingConstituentSeal,
                     "real_composite_identity_claims_incomplete", 0, 0U};
    return result;
  }

  auto& identity = result.audit.identity;
  identity.plan_magic = kSm87BulkV2P40PlanMagic;
  identity.abi_major = kSm87BulkV2P40PlanAbiMajor;
  identity.abi_minor = kSm87BulkV2P40PlanAbiMinor;
  identity.owner_identity = owner.owner_identity_;
  // One owner can be sealed only once.  Its owner-issued nonce is therefore
  // also a unique seal nonce without accepting caller-selected authority.
  identity.seal_nonce = owner.owner_identity_;
  identity.deployment_identity = request.identities.deployment_identity;
  identity.model_identity = request.identities.model_identity;
  identity.request_allocation_identity =
      request.identities.request_allocation_identity;
  identity.stream_event_owner_identity =
      request.identities.stream_event_owner_identity;
  identity.asset_catalog_identity =
      request.identities.asset_catalog_identity;
  identity.binary_evidence_identity =
      request.identities.binary_evidence_identity;
  identity.fp8_oracle_evidence_identity =
      request.identities.fp8_oracle_evidence_identity;
  identity.attention_oracle_evidence_identity =
      request.identities.attention_oracle_evidence_identity;
  identity.gdn_oracle_evidence_identity =
      request.identities.gdn_oracle_evidence_identity;
  identity.nvfp4_oracle_evidence_identity =
      request.identities.nvfp4_oracle_evidence_identity;
  identity.device_ordinal = owner.device_ordinal_;
  identity.execution_class =
      Sm87BulkV2P40ExecutionClass::kDefaultOffDevelopmentCandidate;
  identity.production_dispatch_eligible = false;

  auto& findings = result.audit.constituent_findings;
  findings[constituent_index(Sm87BulkV2P40Constituent::kFp8Projection)] =
      audit_fp8(
          owner.device_ordinal_,
          owner.streams_[static_cast<std::size_t>(
              Sm87BulkV2P40Stream::kProjectionAndGdnProducer)],
          request);
  findings[constituent_index(Sm87BulkV2P40Constituent::kFullAttention)] =
      audit_attention(
          owner.device_ordinal_,
          owner.streams_[static_cast<std::size_t>(Sm87BulkV2P40Stream::kMain)],
          request);
  findings[constituent_index(Sm87BulkV2P40Constituent::kBf16Ab)] =
      request.bf16_ab == nullptr
          ? Sm87BulkV2P40ConstituentFinding::kMissingSealedAccess
          : Sm87BulkV2P40ConstituentFinding::
                kMissingUnforgeableQualificationInterface;
  if (request.gdn == nullptr) {
    findings[constituent_index(Sm87BulkV2P40Constituent::kGdn)] =
        Sm87BulkV2P40ConstituentFinding::kMissingSealedAccess;
  } else if (!q3x::kernels::sm87_bulk_v2_gdn_p40_session_state_valid(
                 *request.gdn)) {
    findings[constituent_index(Sm87BulkV2P40Constituent::kGdn)] =
        Sm87BulkV2P40ConstituentFinding::kInvalidSealedAccess;
  } else {
    findings[constituent_index(Sm87BulkV2P40Constituent::kGdn)] =
        Sm87BulkV2P40ConstituentFinding::kMutableSessionIsNotCapability;
  }
  // Confirmed interface gap: NVFP4 currently exposes a manifest, resource
  // query and admission-only numerical launcher, but no startup-sealed access.
  findings[constituent_index(Sm87BulkV2P40Constituent::kNvFp4Projection)] =
      Sm87BulkV2P40ConstituentFinding::
          kMissingUnforgeableQualificationInterface;
  const auto request_state_finding = audit_request_state(
      owner.owner_identity_, owner.device_ordinal_,
      owner.streams_[static_cast<std::size_t>(
          Sm87BulkV2P40Stream::kMain)],
      request);
  findings[constituent_index(Sm87BulkV2P40Constituent::kRequestArena)] =
      request_state_finding;
  findings[constituent_index(Sm87BulkV2P40Constituent::kPinnedHandoff)] =
      request_state_finding;

  const bool constituents_ready = all_ready(findings);
  result.audit.all_static_resource_checks_complete = constituents_ready;
  result.audit.authenticated_real_constituents = constituents_ready;
  result.audit.exact_numerical_contract_qualified = constituents_ready;
  result.audit.default_off_candidate_eligible = constituents_ready;
  result.audit.production_dispatch_eligible = false;
  result.audit.production_selector_bound = false;
  result.audit.synthetic_host_contract = false;
  result.audit.mtp_used = false;
  result.audit.cublaslt_production_path_used = false;
  result.audit.request_time_jit_used = false;
  result.audit.request_time_repack_used = false;
  result.audit.accuracy_relaxation_used = false;
  identity.authenticated_real_constituents = constituents_ready;
  identity.exact_numerical_contract_qualified = constituents_ready;
  identity.development_execution_eligible = constituents_ready;

  // No current combination can reach this branch until every exact interface
  // above exists and binds its own numerical evidence.  Keeping the branch
  // fail-closed is intentional; manifests and public aggregates are not
  // accepted as shortcuts.
  if (!result.audit.semantics_valid()) {
    const auto blocker = std::find_if(
        findings.begin(), findings.end(), [](const auto finding) noexcept {
          return finding != Sm87BulkV2P40ConstituentFinding::kReady;
        });
    const std::size_t blocker_index =
        blocker == findings.end()
            ? kSm87BulkV2P40ConstituentCount
            : static_cast<std::size_t>(blocker - findings.begin());
    result.status = {Sm87BulkV2P40OwnerError::kMissingConstituentSeal,
                     "real_constituent_capability_or_qualification_missing",
                     0, blocker_index};
    return result;
  }

  result.status = {Sm87BulkV2P40OwnerError::kMissingConstituentSeal,
                   "real_constituent_adapter_not_yet_complete", 0, 0U};
  return result;
}

}  // namespace q3x::runtime::sm87_bulk_v2_p40_owner_detail
