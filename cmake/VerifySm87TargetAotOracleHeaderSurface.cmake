cmake_minimum_required(VERSION 3.24)

foreach(required IN ITEMS
    Q3X_CXX_COMPILER
    Q3X_SOURCE_DIR
    Q3X_BINARY_DIR
    Q3X_CUDA_INCLUDE_DIRS
    Q3X_EXPECT_ORACLE)
  if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
    message(FATAL_ERROR "Missing required header audit input: ${required}")
  endif()
endforeach()

set(audit_directory "${Q3X_BINARY_DIR}/oracle-header-surface-contract")
file(MAKE_DIRECTORY "${audit_directory}")
set(audit_source "${audit_directory}/installed_headers.cpp")
file(WRITE "${audit_source}"
  "#include \"q3x/runtime/model_weights.h\"\n"
  "#include \"q3x/runtime/reference_engine.h\"\n"
  "#include \"q3x/runtime/sm87_target_aot_projection_device_assets.h\"\n")

set(preprocess_command
  "${Q3X_CXX_COMPILER}"
  -E
  -P
  -std=c++17
  -x c++
  "-I${Q3X_SOURCE_DIR}/include"
  "-I${Q3X_BINARY_DIR}/generated")
foreach(cuda_include IN LISTS Q3X_CUDA_INCLUDE_DIRS)
  list(APPEND preprocess_command "-I${cuda_include}")
endforeach()
if(Q3X_EXPECT_ORACLE)
  list(APPEND preprocess_command
    -DQ3X_ENABLE_SM87_TARGET_AOT_LAYER0_M192_ORACLE_ADMISSION=1)
endif()
list(APPEND preprocess_command "${audit_source}")

execute_process(
  COMMAND ${preprocess_command}
  RESULT_VARIABLE preprocess_status
  OUTPUT_VARIABLE preprocessed
  ERROR_VARIABLE preprocess_error
)
if(NOT preprocess_status EQUAL 0)
  message(FATAL_ERROR
    "Installed-header preprocessing failed: ${preprocess_error}")
endif()

set(capability "Sm87TargetAotLayer0M192OracleAccess")
string(REGEX MATCHALL "${capability}" capability_mentions "${preprocessed}")
list(LENGTH capability_mentions capability_mention_count)
string(REGEX REPLACE "[ \t\r\n]+" " " normalized "${preprocessed}")
string(REGEX MATCHALL
  "friend class reference_engine_test_detail:: ${capability}"
  friend_mentions "${normalized}")
list(LENGTH friend_mentions friend_mention_count)

if(Q3X_EXPECT_ORACLE)
  # Each of the three installed headers owns one guarded forward declaration
  # and one guarded friend declaration.
  if(NOT capability_mention_count EQUAL 6 OR
     NOT friend_mention_count EQUAL 3)
    message(FATAL_ERROR
      "Oracle-ON installed-header capability surface is incomplete: "
      "mentions=${capability_mention_count}, friends=${friend_mention_count}")
  endif()
else()
  if(NOT capability_mention_count EQUAL 0 OR
     NOT friend_mention_count EQUAL 0)
    message(FATAL_ERROR
      "Oracle-OFF preprocessed installed headers expose the private "
      "capability: mentions=${capability_mention_count}, "
      "friends=${friend_mention_count}")
  endif()
endif()

message(STATUS
  "SM87 target-AOT installed-header Oracle surface matches admission=${Q3X_EXPECT_ORACLE}")
