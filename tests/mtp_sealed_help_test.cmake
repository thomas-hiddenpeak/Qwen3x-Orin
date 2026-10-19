if(NOT DEFINED Q3X_EVAL_SERVER OR NOT Q3X_MTP_CAP MATCHES "^[23]$")
  message(FATAL_ERROR "MTP sealed help test requires server and cap 2/3")
endif()
execute_process(COMMAND "${Q3X_EVAL_SERVER}" --help
  RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "MTP help failed: ${result}: ${error}")
endif()
foreach(required IN ITEMS
    "q3x.sm87.candidate.mtp-eos-frontier-cap${Q3X_MTP_CAP}.v122"
    "MTP draft cap: ${Q3X_MTP_CAP} (compiled)"
    "candidate, not production qualified or installable"
    "Fixed concurrency: one GPU request, one queued request, three HTTP threads")
  string(FIND "${output}" "${required}" position)
  if(position EQUAL -1)
    message(FATAL_ERROR "Missing sealed MTP contract: ${required}")
  endif()
endforeach()
foreach(forbidden IN ITEMS "--candidate-profile" "--production-profile"
    "--development-route" "--prefill-projection-tactic" "--projection-backend"
    "Q3X_MTP_DRAFT_LENGTH" "--queue-capacity" "--ingress-threads")
  string(FIND "${output}" "${forbidden}" position)
  if(NOT position EQUAL -1)
    message(FATAL_ERROR "Sealed MTP help exposes ${forbidden}")
  endif()
endforeach()
