cmake_minimum_required(VERSION 3.24)

foreach(required IN ITEMS Q3X_PROBE Q3X_NM Q3X_EXPECT_ORACLE)
  if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
    message(FATAL_ERROR "Missing required probe audit input: ${required}")
  endif()
endforeach()
if(NOT EXISTS "${Q3X_PROBE}")
  message(FATAL_ERROR "Prepare probe does not exist: ${Q3X_PROBE}")
endif()

file(STRINGS "${Q3X_PROBE}" contract_strings
  REGEX "schema_version|claim_boundary|execution_identity|layer0_m192_oracle")
string(JOIN "\n" contract_surface ${contract_strings})
execute_process(
  COMMAND "${Q3X_NM}" -C "${Q3X_PROBE}"
  RESULT_VARIABLE nm_status
  OUTPUT_VARIABLE nm_output
  ERROR_VARIABLE nm_error
)
if(NOT nm_status EQUAL 0)
  message(FATAL_ERROR "nm failed for ${Q3X_PROBE}: ${nm_error}")
endif()

set(schema_v2 "\"schema_version\": 2")
set(schema_v3 "\"schema_version\": 3")
set(v2_claim
  "and destruction only; no launcher, generation, numerical, performance, or production-route authority")
set(v3_claim
  "fixed-M192 layer-0 numerical/resource oracle, and destruction only")
set(oracle_symbol
  "Sm87TargetAotLayer0M192OracleAccess::screen")

if(Q3X_EXPECT_ORACLE)
  foreach(required_text IN ITEMS
      "${schema_v3}"
      "${v3_claim}"
      "execution_identity"
      "target_layer0_m192_oracle"
      "layer0_m192_oracle_passed")
    string(FIND "${contract_surface}" "${required_text}" found)
    if(found EQUAL -1)
      message(FATAL_ERROR
        "Oracle-ON prepare probe is missing v3 contract text: ${required_text}")
    endif()
  endforeach()
  string(FIND "${contract_surface}" "${schema_v2}" forbidden_v2)
  string(FIND "${nm_output}" "${oracle_symbol}" oracle_symbol_index)
  if(NOT forbidden_v2 EQUAL -1 OR oracle_symbol_index EQUAL -1)
    message(FATAL_ERROR
      "Oracle-ON prepare probe did not isolate the v3 schema/Oracle gate")
  endif()
else()
  foreach(required_text IN ITEMS "${schema_v2}" "${v2_claim}")
    string(FIND "${contract_surface}" "${required_text}" found)
    if(found EQUAL -1)
      message(FATAL_ERROR
        "Oracle-OFF prepare probe is missing the v2 contract: ${required_text}")
    endif()
  endforeach()
  foreach(forbidden_text IN ITEMS
      "${schema_v3}"
      "execution_identity"
      "layer0_m192_oracle")
    string(FIND "${contract_surface}" "${forbidden_text}" found)
    if(NOT found EQUAL -1)
      message(FATAL_ERROR
        "Oracle-OFF prepare probe exposes v3/Oracle text: ${forbidden_text}")
    endif()
  endforeach()
  string(FIND "${nm_output}" "${oracle_symbol}" oracle_symbol_index)
  if(NOT oracle_symbol_index EQUAL -1)
    message(FATAL_ERROR
      "Oracle-OFF prepare probe links the private Oracle gate")
  endif()
endif()

message(STATUS
  "SM87 target-AOT prepare probe matches admission=${Q3X_EXPECT_ORACLE}")
