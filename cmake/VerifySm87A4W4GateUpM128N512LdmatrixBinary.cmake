if(NOT DEFINED Q3X_CUOBJDUMP_EXECUTABLE OR
   NOT EXISTS "${Q3X_CUOBJDUMP_EXECUTABLE}")
  message(FATAL_ERROR "cuobjdump is unavailable for M128N512 LDSM audit")
endif()
if(NOT DEFINED Q3X_M128N512_LDMATRIX_BINARY OR
   NOT EXISTS "${Q3X_M128N512_LDMATRIX_BINARY}")
  message(FATAL_ERROR "M128N512 LDSM correctness binary is missing")
endif()
if(NOT DEFINED Q3X_M128N512_LDMATRIX_SOURCE OR
   NOT EXISTS "${Q3X_M128N512_LDMATRIX_SOURCE}")
  message(FATAL_ERROR "M128N512 LDSM kernel source is missing")
endif()
if(NOT DEFINED Q3X_M128N512_LDMATRIX_AUDIT_DIR)
  set(Q3X_M128N512_LDMATRIX_AUDIT_DIR
      "${CMAKE_CURRENT_BINARY_DIR}/sm87_a4w4_m128n512_ldmatrix_audit")
endif()
file(MAKE_DIRECTORY "${Q3X_M128N512_LDMATRIX_AUDIT_DIR}")

# The whole kernel legitimately contains scalar shared loads for K512 scales
# and quantization.  Reject the legacy scalar A-fragment helper at source and
# then lock the compiled MMA feed by exact LDSM.x4/IMMA counts.
file(READ "${Q3X_M128N512_LDMATRIX_SOURCE}" _q3x_source)
string(FIND "${_q3x_source}"
  "sm87_a4w4_load_a_fragment_swizzled_shared" _q3x_scalar_a_helper)
if(NOT _q3x_scalar_a_helper EQUAL -1)
  message(FATAL_ERROR "M128N512 main feed calls the scalar-A LDS helper")
endif()

execute_process(
  COMMAND "${Q3X_CUOBJDUMP_EXECUTABLE}" --dump-sass
          "${Q3X_M128N512_LDMATRIX_BINARY}"
  RESULT_VARIABLE _q3x_sass_result
  OUTPUT_VARIABLE _q3x_sass
  ERROR_VARIABLE _q3x_sass_error
)
if(NOT _q3x_sass_result EQUAL 0)
  message(FATAL_ERROR "M128N512 LDSM SASS dump failed: ${_q3x_sass_error}")
endif()
set(_q3x_symbol
    "q3x_sm87_a4w4_gateup_down_k512_edge_m128n512_ldmatrix_kernel")
set(_q3x_marker "Function : ${_q3x_symbol}")
string(FIND "${_q3x_sass}" "${_q3x_marker}" _q3x_start)
if(_q3x_start EQUAL -1)
  message(FATAL_ERROR "M128N512 LDSM SASS symbol is missing")
endif()
string(SUBSTRING "${_q3x_sass}" ${_q3x_start} -1 _q3x_kernel)
string(FIND "${_q3x_kernel}" "\n\t\tFunction : " _q3x_next)
if(NOT _q3x_next EQUAL -1)
  string(SUBSTRING "${_q3x_kernel}" 0 ${_q3x_next} _q3x_kernel)
endif()
file(WRITE "${Q3X_M128N512_LDMATRIX_AUDIT_DIR}/kernel.sass"
     "${_q3x_kernel}")

string(REGEX MATCHALL "LDSM[.]16[.]M88[.]4[ \t]" _q3x_x4
  "${_q3x_kernel}")
string(REGEX MATCHALL "LDSM[.]16[.]M88[.](1|2)[ \t]" _q3x_other_ldsm
  "${_q3x_kernel}")
string(REGEX MATCHALL "IMMA[.]16864[.]S4[.]S4[ \t]" _q3x_imma
  "${_q3x_kernel}")
list(LENGTH _q3x_x4 _q3x_x4_count)
list(LENGTH _q3x_other_ldsm _q3x_other_ldsm_count)
list(LENGTH _q3x_imma _q3x_imma_count)
if(NOT _q3x_x4_count EQUAL 64 OR
   NOT _q3x_other_ldsm_count EQUAL 0 OR
   NOT _q3x_imma_count EQUAL 128)
  message(FATAL_ERROR
    "M128N512 operand feed failed: x4=${_q3x_x4_count} (expected 64), "
    "other-LDSM=${_q3x_other_ldsm_count} (expected 0), "
    "IMMA=${_q3x_imma_count} (expected 128)"
  )
endif()
foreach(_q3x_forbidden IN ITEMS LDL STL)
  string(REGEX MATCHALL
    "[ \t]${_q3x_forbidden}([.][A-Za-z0-9]+)*[ \t]"
    _q3x_forbidden_matches "${_q3x_kernel}")
  list(LENGTH _q3x_forbidden_matches _q3x_forbidden_count)
  if(NOT _q3x_forbidden_count EQUAL 0)
    message(FATAL_ERROR
      "M128N512 kernel emitted ${_q3x_forbidden_count} ${_q3x_forbidden} instruction(s)"
    )
  endif()
endforeach()

execute_process(
  COMMAND "${Q3X_CUOBJDUMP_EXECUTABLE}" --dump-resource-usage
          "${Q3X_M128N512_LDMATRIX_BINARY}"
  RESULT_VARIABLE _q3x_resource_result
  OUTPUT_VARIABLE _q3x_resources
  ERROR_VARIABLE _q3x_resource_error
)
if(NOT _q3x_resource_result EQUAL 0)
  message(FATAL_ERROR
    "M128N512 LDSM resource dump failed: ${_q3x_resource_error}"
  )
endif()
file(WRITE "${Q3X_M128N512_LDMATRIX_AUDIT_DIR}/kernel.resources"
     "${_q3x_resources}")
set(_q3x_resource_marker "Function ${_q3x_symbol}:")
string(FIND "${_q3x_resources}" "${_q3x_resource_marker}" _q3x_resource_start)
if(_q3x_resource_start EQUAL -1)
  message(FATAL_ERROR "M128N512 LDSM resource record is missing")
endif()
string(SUBSTRING "${_q3x_resources}" ${_q3x_resource_start} -1
       _q3x_resource_tail)
string(REGEX MATCH
  "REG:[0-9]+ STACK:[0-9]+ SHARED:[0-9]+ LOCAL:[0-9]+"
  _q3x_record "${_q3x_resource_tail}")
if(NOT _q3x_record)
  message(FATAL_ERROR "M128N512 LDSM resource values are missing")
endif()
string(REGEX REPLACE ".*REG:([0-9]+).*" "\\1" _q3x_registers
  "${_q3x_record}")
string(REGEX REPLACE ".*STACK:([0-9]+).*" "\\1" _q3x_stack
  "${_q3x_record}")
string(REGEX REPLACE ".*SHARED:([0-9]+).*" "\\1" _q3x_shared
  "${_q3x_record}")
string(REGEX REPLACE ".*LOCAL:([0-9]+).*" "\\1" _q3x_local
  "${_q3x_record}")
if(_q3x_registers GREATER 255 OR _q3x_registers LESS 1 OR
   NOT _q3x_stack EQUAL 0 OR NOT _q3x_shared EQUAL 0 OR
   NOT _q3x_local EQUAL 0)
  message(FATAL_ERROR "M128N512 resource contract failed: ${_q3x_record}")
endif()

message(STATUS
  "Verified M128N512 LDSM feed: 64 x4, 128 IMMA, no scalar-A helper, "
  "LDL/STL=0, ${_q3x_record}"
)
