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

set(required_fragments
  "q3x.sm87.production.p40.legacy-c512-exact.v2"
  "P40000 prompt, 4096 output ceiling"
  "44095 resident Legacy-C512/SM87 capacity"
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
foreach(fragment IN LISTS forbidden_fragments)
  string(FIND "${output}" "${fragment}" position)
  if(NOT position EQUAL -1)
    message(FATAL_ERROR
      "production help exposes sealed selector '${fragment}'\n${output}")
  endif()
endforeach()
