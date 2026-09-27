if(NOT DEFINED Q3X_EVAL_SERVER)
  message(FATAL_ERROR "production help test is missing Q3X_EVAL_SERVER")
endif()

execute_process(
  COMMAND "${Q3X_EVAL_SERVER}" --help
  RESULT_VARIABLE result
  OUTPUT_VARIABLE output
  ERROR_VARIABLE error
)
if(NOT result EQUAL 0)
  message(FATAL_ERROR
    "qwen3x-eval-server --help failed (${result})\n${output}${error}")
endif()

if(NOT DEFINED Q3X_EXPECT_PROFILE)
  # BUILD_TESTING=ON retains the legacy profile. Installed Release/OFF checks
  # must explicitly request whole-core-service so a stale binary cannot pass.
  set(Q3X_EXPECT_PROFILE "legacy")
endif()
if(Q3X_EXPECT_PROFILE STREQUAL "whole-core-service")
  set(required_fragments
    "q3x.sm87.production.whole-core-service.v1"
    "Prompt + output - 1 <= 44095; output ceiling 4096; greedy text/chat"
    "Fixed concurrency: one GPU request, one queued request, three HTTP threads"
  )
elseif(Q3X_EXPECT_PROFILE STREQUAL "legacy")
  set(required_fragments
    "q3x.sm87.candidate.p40.legacy-c512-terminal-prefix.v1"
    "Engineering candidate; not production/release qualified"
    "P40000 prompt, 4096 output ceiling"
    "44095 resident Legacy-C512/SM87 capacity"
  )
else()
  message(FATAL_ERROR "unknown Q3X_EXPECT_PROFILE: ${Q3X_EXPECT_PROFILE}")
endif()
list(APPEND required_fragments
  "--api-key-file PATH"
  "/healthz remains"
)
foreach(fragment IN LISTS required_fragments)
  string(FIND "${output}" "${fragment}" position)
  if(position EQUAL -1)
    message(FATAL_ERROR
      "production help is missing '${fragment}'\n${output}")
  endif()
endforeach()

set(forbidden_fragments
  "--development-route"
  "--candidate-profile"
  "--production-profile"
  "--max-sequence-length"
  "--max-output-tokens"
  "--prefill-chunk-size"
  "--prefill-execution-mode"
  "--prefill-attention-tactic"
  "--prefill-projection-tactic"
  "--projection-backend"
  "--request-max-arena-bytes"
  "--min-free-bytes"
  "unauthenticated evaluation surface"
  "not a production serving API"
)
if(Q3X_EXPECT_PROFILE STREQUAL "whole-core-service")
  list(APPEND forbidden_fragments
    "Legacy-C512"
    "--queue-capacity"
    "--ingress-threads"
  )
endif()
foreach(fragment IN LISTS forbidden_fragments)
  string(FIND "${output}" "${fragment}" position)
  if(NOT position EQUAL -1)
    message(FATAL_ERROR
      "production help exposes sealed selector '${fragment}'\n${output}")
  endif()
endforeach()
