cmake_minimum_required(VERSION 3.24)

foreach(required IN ITEMS
    Q3X_PROBE Q3X_NM Q3X_CTEST Q3X_BUILD_DIR Q3X_SOURCE_DIR)
  if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
    message(FATAL_ERROR
      "Missing required MacroFeed V4 lifetime probe input: ${required}")
  endif()
endforeach()
if(NOT EXISTS "${Q3X_PROBE}")
  message(FATAL_ERROR "Lifetime probe does not exist: ${Q3X_PROBE}")
endif()
string(RANDOM LENGTH 32 ALPHABET 0123456789abcdef contract_nonce)

file(STRINGS "${Q3X_PROBE}" contract_strings
  REGEX
    "schema_version|claim_boundary|checkpoint_identity|resident_shards_exact|complete-GDN|MLP-pair|Full-Attention|macrofeed_v4_normal_catalogs|complete_gdn_catalog_identity|complete_gdn_binding_count|mlp_pair_catalog_identity|mlp_pair_binding_count|full_attention_catalog_identity|full_attention_binding_count|retained_complete_gdn_catalog_fold_identity|retained_mlp_pair_catalog_fold_identity|retained_full_attention_catalog_fold_identity|full_attention_resource_bundle_identity|full_attention_owner|kv_allocation_identity|request_state_kv_allocation_identity|kv_allocation_bytes|engine_rope_owner_identity|engine_rope_binding_identity|engine_rope_allocation_bytes|execution_owned_bytes|rope_anchored_bytes|aggregate_reserve_chain|minimum_free_bytes_after_|execution_required_device_allocation_bytes|execution_aggregate_memory_gate_passed|owner_allocation_device_lifetime_chain|lifetime_root_identity|receipt_schema|receipt_sha256|self_sha256|flags|admissions|macrofeed_v4_full_attention_preprocess|macrofeed_v4_attention_c8000|selector_bound|api_route_bound")
string(JOIN "\n" contract_surface ${contract_strings})
foreach(required_text IN ITEMS
    "\"schema_version\": 4"
    "normal 48-complete-GDN, 64-MLP-pair, and 16-Full-Attention"
    "exact V4 KV and shared Engine RoPE ownership"
    "staged aggregate reserve arguments"
    "32 MiB tolerance"
    "checkpoint_identity"
    "resident_shards_exact"
    "macrofeed_v4_normal_catalogs"
    "complete_gdn_catalog_identity"
    "complete_gdn_binding_count"
    "mlp_pair_catalog_identity"
    "mlp_pair_binding_count"
    "full_attention_catalog_identity"
    "full_attention_binding_count"
    "retained_complete_gdn_catalog_fold_identity"
    "retained_mlp_pair_catalog_fold_identity"
    "retained_full_attention_catalog_fold_identity"
    "full_attention_resource_bundle_identity"
    "full_attention_owner"
    "kv_allocation_identity"
    "request_state_kv_allocation_identity"
    "kv_allocation_bytes"
    "engine_rope_owner_identity"
    "engine_rope_binding_identity"
    "engine_rope_allocation_bytes"
    "execution_owned_bytes"
    "rope_anchored_bytes"
    "aggregate_reserve_chain"
    "minimum_free_bytes_after_legacy_create"
    "minimum_free_bytes_after_execution_create"
    "minimum_free_bytes_after_rope_create"
    "minimum_free_bytes_after_complete_aot_create"
    "execution_required_device_allocation_bytes"
    "execution_aggregate_memory_gate_passed"
    "owner_allocation_device_lifetime_chain"
    "lifetime_root_identity"
    "receipt_schema"
    "receipt_sha256"
    "self_sha256"
    "flags"
    "admissions"
    "macrofeed_v4_full_attention_preprocess"
    "macrofeed_v4_attention_c8000"
    "selector_bound"
    "api_route_bound")
  string(FIND "${contract_surface}" "${required_text}" found)
  if(found EQUAL -1)
    message(FATAL_ERROR
      "Lifetime probe is missing fail-closed JSON field: ${required_text}")
  endif()
endforeach()

# The executable surface proves that evidence is emitted.  The private source
# contract additionally pins the arithmetic values which are encoded as
# machine immediates rather than necessarily surviving as binary strings.
set(snapshot_header
  "${Q3X_SOURCE_DIR}/src/runtime/sm87_macrofeed_v4_engine_lifetime_probe_internal.h")
set(probe_source
  "${Q3X_SOURCE_DIR}/tests/sm87_macrofeed_v4_real_checkpoint_engine_lifetime_probe.cpp")
foreach(required_source IN ITEMS "${snapshot_header}" "${probe_source}")
  if(NOT EXISTS "${required_source}")
    message(FATAL_ERROR
      "Lifetime probe private contract source is missing: ${required_source}")
  endif()
endforeach()
file(READ "${snapshot_header}" snapshot_contract_source)
file(READ "${probe_source}" probe_contract_source)
foreach(required_text IN ITEMS
    "kSm87MacroFeedV4EngineLifetimeExpectedFullAttentionBindings = 16U"
    "kSm87MacroFeedV4EngineLifetimeExpectedOwnedBytes = 3'220'701'184U"
    "kSm87MacroFeedV4EngineLifetimeExpectedKvArenaBytes = 2'621'440'000U"
    "kSm87MacroFeedV4EngineLifetimeExpectedRopeBytes = 67'108'864U"
    "kSm87MacroFeedV4EngineLifetimeExpectedAnchoredBytes = 3'287'810'048U"
    "kSm87MacroFeedV4EngineLifetimeLegacyRequestArenaBytes = 8'640'542'976U"
    "11'928'353'024U")
  string(FIND "${snapshot_contract_source}" "${required_text}" found)
  if(found EQUAL -1)
    message(FATAL_ERROR
      "Lifetime probe private snapshot contract is missing: ${required_text}")
  endif()
endforeach()
foreach(required_text IN ITEMS
    "32ULL * 1024ULL * 1024ULL"
    "kDestroyRecoveryToleranceBytes <"
    "evidence.full_attention_ownership_exact"
    "evidence.reserve_chain_exact")
  string(FIND "${probe_contract_source}" "${required_text}" found)
  if(found EQUAL -1)
    message(FATAL_ERROR
      "Lifetime probe implementation contract is missing: ${required_text}")
  endif()
endforeach()

execute_process(
  COMMAND "${Q3X_NM}" -C "${Q3X_PROBE}"
  RESULT_VARIABLE nm_status
  OUTPUT_VARIABLE nm_output
  ERROR_VARIABLE nm_error
)
if(NOT nm_status EQUAL 0)
  message(FATAL_ERROR "nm failed for ${Q3X_PROBE}: ${nm_error}")
endif()
foreach(required_symbol IN ITEMS
    "exchange_sm87_macrofeed_v4_engine_lifetime_construction_snapshot_hook"
    "Sm87MacroFeedV4P40ExecutionCompositionRoot::construction_snapshot")
  string(FIND "${nm_output}" "${required_symbol}" found)
  if(found EQUAL -1)
    message(FATAL_ERROR
      "Lifetime probe is missing its private observation seam: ${required_symbol}")
  endif()
endforeach()
string(FIND "${nm_output}" "create_with_synthetic_t1" synthetic_wrapper)
if(NOT synthetic_wrapper EQUAL -1)
  message(FATAL_ERROR
    "Lifetime probe unexpectedly exports the named synthetic-T1 wrapper")
endif()

# EXCLUDE_FROM_ALL is not enough on its own: this real-checkpoint executable
# must also stay outside CTest so ordinary test runs cannot allocate the full
# model/device owner accidentally.
execute_process(
  COMMAND "${Q3X_CTEST}" --test-dir "${Q3X_BUILD_DIR}" -N
  RESULT_VARIABLE ctest_status
  OUTPUT_VARIABLE ctest_output
  ERROR_VARIABLE ctest_error
)
if(NOT ctest_status EQUAL 0)
  message(FATAL_ERROR "ctest -N failed: ${ctest_error}")
endif()
string(FIND "${ctest_output}"
  "q3x_sm87_macrofeed_v4_real_checkpoint_engine_lifetime_probe"
  registered_probe)
if(NOT registered_probe EQUAL -1)
  message(FATAL_ERROR
    "Real-checkpoint lifetime probe must not be registered with CTest")
endif()

# Every invocation below fails during argument/path admission, before the
# probe queries CUDA or opens the model.  It therefore checks the create-only
# repository-local evidence boundary without becoming a device run.
execute_process(
  COMMAND "${Q3X_PROBE}"
  RESULT_VARIABLE missing_arguments_status
  OUTPUT_QUIET
  ERROR_QUIET
)
if(NOT missing_arguments_status EQUAL 2)
  message(FATAL_ERROR
    "Lifetime probe did not reject missing arguments with exit 2")
endif()

set(forbidden_output
  "${Q3X_SOURCE_DIR}/q3x-v4-lifetime-probe-forbidden-${contract_nonce}.json")
if(EXISTS "${forbidden_output}")
  message(FATAL_ERROR
    "Unique forbidden-output fixture unexpectedly already exists: ${forbidden_output}")
endif()
execute_process(
  COMMAND "${Q3X_PROBE}" ignored-model "${forbidden_output}"
  RESULT_VARIABLE forbidden_output_status
  OUTPUT_QUIET
  ERROR_QUIET
)
if(NOT forbidden_output_status EQUAL 2 OR EXISTS "${forbidden_output}")
  # The randomized path was proven absent immediately before this invocation,
  # so an unexpected file here was created by the probe under test rather than
  # supplied by the workspace owner.
  if(EXISTS "${forbidden_output}")
    file(REMOVE "${forbidden_output}")
  endif()
  message(FATAL_ERROR
    "Lifetime probe did not fail closed on an output outside .q3x-work")
endif()

set(contract_directory
  "${Q3X_BUILD_DIR}/generated/tests/lifetime-probe-contract-${contract_nonce}")
if(EXISTS "${contract_directory}")
  message(FATAL_ERROR
    "Unique lifetime-probe contract directory unexpectedly exists: ${contract_directory}")
endif()
file(MAKE_DIRECTORY "${contract_directory}")
set(extensionless_output "${contract_directory}/missing-extension")
execute_process(
  COMMAND "${Q3X_PROBE}" ignored-model "${extensionless_output}"
  RESULT_VARIABLE extensionless_status
  OUTPUT_QUIET
  ERROR_QUIET
)
if(NOT extensionless_status EQUAL 2 OR EXISTS "${extensionless_output}")
  if(EXISTS "${extensionless_output}")
    file(REMOVE "${extensionless_output}")
  endif()
  file(REMOVE_RECURSE "${contract_directory}")
  message(FATAL_ERROR
    "Lifetime probe did not reject an extensionless evidence path")
endif()

set(existing_output "${contract_directory}/existing.json")
file(WRITE "${existing_output}" "owner-data\n")
execute_process(
  COMMAND "${Q3X_PROBE}" ignored-model "${existing_output}"
  RESULT_VARIABLE existing_output_status
  OUTPUT_QUIET
  ERROR_QUIET
)
file(READ "${existing_output}" existing_output_contents)
if(NOT existing_output_status EQUAL 2 OR
   NOT existing_output_contents STREQUAL "owner-data\n")
  file(REMOVE_RECURSE "${contract_directory}")
  message(FATAL_ERROR
    "Lifetime probe did not preserve an existing evidence file")
endif()
file(REMOVE_RECURSE "${contract_directory}")

message(STATUS
  "MacroFeed V4 lifetime probe static/path contract passes")
