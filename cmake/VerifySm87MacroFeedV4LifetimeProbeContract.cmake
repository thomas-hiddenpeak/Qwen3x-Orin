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
    "schema_version|claim_boundary|checkpoint_identity|resident_shards_exact|complete-GDN|MLP-pair|macrofeed_v4_normal_catalogs|complete_gdn_catalog_identity|complete_gdn_binding_count|mlp_pair_catalog_identity|mlp_pair_binding_count|retained_complete_gdn_catalog_fold_identity|retained_mlp_pair_catalog_fold_identity|owner_allocation_device_lifetime_chain|lifetime_root_identity|receipt_schema|receipt_sha256|self_sha256|flags|admissions|selector_bound|api_route_bound")
string(JOIN "\n" contract_surface ${contract_strings})
foreach(required_text IN ITEMS
    "\"schema_version\": 3"
    "normal 48-complete-GDN and "
    "64-MLP-pair catalog identities/folds"
    "checkpoint_identity"
    "resident_shards_exact"
    "macrofeed_v4_normal_catalogs"
    "complete_gdn_catalog_identity"
    "complete_gdn_binding_count"
    "mlp_pair_catalog_identity"
    "mlp_pair_binding_count"
    "retained_complete_gdn_catalog_fold_identity"
    "retained_mlp_pair_catalog_fold_identity"
    "owner_allocation_device_lifetime_chain"
    "lifetime_root_identity"
    "receipt_schema"
    "receipt_sha256"
    "self_sha256"
    "flags"
    "admissions"
    "selector_bound"
    "api_route_bound")
  string(FIND "${contract_surface}" "${required_text}" found)
  if(found EQUAL -1)
    message(FATAL_ERROR
      "Lifetime probe is missing fail-closed JSON field: ${required_text}")
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
