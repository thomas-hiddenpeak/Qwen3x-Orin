if(NOT DEFINED Q3X_CUOBJDUMP_EXECUTABLE OR
   NOT EXISTS "${Q3X_CUOBJDUMP_EXECUTABLE}")
  message(FATAL_ERROR "cuobjdump is unavailable for the attention resource audit")
endif()
if(NOT DEFINED Q3X_ATTENTION_SUPERMATRIX_BINARY OR
   NOT EXISTS "${Q3X_ATTENTION_SUPERMATRIX_BINARY}")
  message(FATAL_ERROR "attention supermatrix binary is missing")
endif()

execute_process(
  COMMAND "${Q3X_CUOBJDUMP_EXECUTABLE}" --dump-resource-usage
          "${Q3X_ATTENTION_SUPERMATRIX_BINARY}"
  RESULT_VARIABLE _q3x_result
  OUTPUT_VARIABLE _q3x_resources
  ERROR_VARIABLE _q3x_error
)
if(NOT _q3x_result EQUAL 0)
  message(FATAL_ERROR "cuobjdump resource audit failed: ${_q3x_error}")
endif()

string(REGEX MATCHALL
  "REG:[0-9]+ STACK:[0-9]+ SHARED:[0-9]+ LOCAL:[0-9]+"
  _q3x_resource_lines "${_q3x_resources}"
)
list(LENGTH _q3x_resource_lines _q3x_resource_count)
if(NOT _q3x_resource_count EQUAL 6)
  message(FATAL_ERROR
    "expected real and small-K Linear/Full/O kernels; found ${_q3x_resource_count}"
  )
endif()

foreach(_q3x_line IN LISTS _q3x_resource_lines)
  string(REGEX REPLACE ".*REG:([0-9]+).*" "\\1" _q3x_registers
                       "${_q3x_line}")
  string(REGEX REPLACE ".*STACK:([0-9]+).*" "\\1" _q3x_stack
                       "${_q3x_line}")
  string(REGEX REPLACE ".*SHARED:([0-9]+).*" "\\1" _q3x_shared
                       "${_q3x_line}")
  string(REGEX REPLACE ".*LOCAL:([0-9]+).*" "\\1" _q3x_local
                       "${_q3x_line}")
  if(_q3x_registers GREATER 128 OR NOT _q3x_stack EQUAL 0 OR
     NOT _q3x_shared EQUAL 42240 OR NOT _q3x_local EQUAL 0)
    message(FATAL_ERROR "attention resource gate failed: ${_q3x_line}")
  endif()
endforeach()

message(STATUS
  "Verified six attention supermatrix kernels: <=128 registers, stack/local 0, shared 42240"
)
