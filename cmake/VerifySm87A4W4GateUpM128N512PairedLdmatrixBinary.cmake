if(NOT DEFINED Q3X_CUOBJDUMP_EXECUTABLE OR
   NOT EXISTS "${Q3X_CUOBJDUMP_EXECUTABLE}")
  message(FATAL_ERROR
    "cuobjdump is unavailable for the paired M128N512 LDSM audit")
endif()
if(NOT DEFINED Q3X_M128N512_PAIRED_LDMATRIX_BINARY OR
   NOT EXISTS "${Q3X_M128N512_PAIRED_LDMATRIX_BINARY}")
  message(FATAL_ERROR
    "paired M128N512 LDSM correctness binary is missing")
endif()
if(NOT DEFINED Q3X_M128N512_PAIRED_LDMATRIX_SOURCE OR
   NOT EXISTS "${Q3X_M128N512_PAIRED_LDMATRIX_SOURCE}")
  message(FATAL_ERROR "paired M128N512 LDSM kernel source is missing")
endif()
if(NOT DEFINED Q3X_M128N512_PAIRED_LDMATRIX_AUDIT_DIR)
  set(Q3X_M128N512_PAIRED_LDMATRIX_AUDIT_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/sm87_a4w4_m128n512_paired_ldmatrix_audit")
endif()
file(MAKE_DIRECTORY "${Q3X_M128N512_PAIRED_LDMATRIX_AUDIT_DIR}")

# The kernel legitimately uses scalar shared loads for scales and the output
# quantizer.  Keep the A feed on the verified x4 mapping at source level, then
# lock the compiled A/B/MMA/pipeline skeleton by exact instruction counts.
file(READ "${Q3X_M128N512_PAIRED_LDMATRIX_SOURCE}" _q3x_source)
string(FIND "${_q3x_source}"
  "sm87_a4w4_load_a_fragment_swizzled_shared" _q3x_scalar_a_helper)
if(NOT _q3x_scalar_a_helper EQUAL -1)
  message(FATAL_ERROR
    "paired M128N512 main feed calls the scalar-A LDS helper")
endif()

execute_process(
  COMMAND "${Q3X_CUOBJDUMP_EXECUTABLE}" --dump-sass
          "${Q3X_M128N512_PAIRED_LDMATRIX_BINARY}"
  RESULT_VARIABLE _q3x_sass_result
  OUTPUT_VARIABLE _q3x_sass
  ERROR_VARIABLE _q3x_sass_error
)
if(NOT _q3x_sass_result EQUAL 0)
  message(FATAL_ERROR
    "paired M128N512 LDSM SASS dump failed: ${_q3x_sass_error}")
endif()
set(_q3x_symbol
  "q3x_sm87_a4w4_gateup_down_k512_edge_m128n512_paired_ldmatrix_kernel")
set(_q3x_marker "Function : ${_q3x_symbol}")
string(FIND "${_q3x_sass}" "${_q3x_marker}" _q3x_start)
if(_q3x_start EQUAL -1)
  message(FATAL_ERROR "paired M128N512 LDSM SASS symbol is missing")
endif()
string(SUBSTRING "${_q3x_sass}" ${_q3x_start} -1 _q3x_kernel)
string(FIND "${_q3x_kernel}" "\n\t\tFunction : " _q3x_next)
if(NOT _q3x_next EQUAL -1)
  string(SUBSTRING "${_q3x_kernel}" 0 ${_q3x_next} _q3x_kernel)
endif()
file(WRITE
  "${Q3X_M128N512_PAIRED_LDMATRIX_AUDIT_DIR}/kernel.sass"
  "${_q3x_kernel}")

function(q3x_count_matches input regex output)
  string(REGEX MATCHALL "${regex}" _q3x_matches "${input}")
  list(LENGTH _q3x_matches _q3x_count)
  set(${output} "${_q3x_count}" PARENT_SCOPE)
endfunction()

q3x_count_matches("${_q3x_kernel}" "LDSM[.]16[.]M88[.]4[ \t]"
  _q3x_ldsm_x4)
q3x_count_matches("${_q3x_kernel}" "LDSM[.]16[.]M88[.](1|2)[ \t]"
  _q3x_other_ldsm)
q3x_count_matches("${_q3x_kernel}" "[ \t]LDS[.]128[ \t]"
  _q3x_paired_b_lds128)
q3x_count_matches("${_q3x_kernel}" "IMMA[.]16864[.]S4[.]S4[ \t]"
  _q3x_imma)
q3x_count_matches("${_q3x_kernel}" "[ \t]BAR[.]SYNC([.][A-Z_]+)*[ \t]"
  _q3x_bar_sites)
q3x_count_matches("${_q3x_kernel}" "[ \t]LDGSTS([.][A-Z0-9_]+)*[ \t]"
  _q3x_ldgsts)
if(NOT _q3x_ldsm_x4 EQUAL 32 OR
   NOT _q3x_other_ldsm EQUAL 0 OR
   NOT _q3x_paired_b_lds128 EQUAL 8 OR
   NOT _q3x_imma EQUAL 64 OR
   NOT _q3x_bar_sites EQUAL 7 OR
   NOT _q3x_ldgsts EQUAL 20)
  message(FATAL_ERROR
    "paired M128N512 instruction contract failed: "
    "x4=${_q3x_ldsm_x4}/32 other-LDSM=${_q3x_other_ldsm}/0 "
    "paired-B-LDS.128=${_q3x_paired_b_lds128}/8 "
    "IMMA=${_q3x_imma}/64 BAR-sites=${_q3x_bar_sites}/7 "
    "LDGSTS=${_q3x_ldgsts}/20")
endif()
foreach(_q3x_forbidden IN ITEMS LDL STL)
  q3x_count_matches("${_q3x_kernel}"
    "[ \t]${_q3x_forbidden}([.][A-Za-z0-9_]+)*[ \t]"
    _q3x_forbidden_count)
  if(NOT _q3x_forbidden_count EQUAL 0)
    message(FATAL_ERROR
      "paired M128N512 kernel emitted ${_q3x_forbidden_count} "
      "${_q3x_forbidden} instruction(s)")
  endif()
endforeach()

execute_process(
  COMMAND "${Q3X_CUOBJDUMP_EXECUTABLE}" --dump-resource-usage
          "${Q3X_M128N512_PAIRED_LDMATRIX_BINARY}"
  RESULT_VARIABLE _q3x_resource_result
  OUTPUT_VARIABLE _q3x_resources
  ERROR_VARIABLE _q3x_resource_error
)
if(NOT _q3x_resource_result EQUAL 0)
  message(FATAL_ERROR
    "paired M128N512 resource dump failed: ${_q3x_resource_error}")
endif()
file(WRITE
  "${Q3X_M128N512_PAIRED_LDMATRIX_AUDIT_DIR}/binary.resources"
  "${_q3x_resources}")
set(_q3x_resource_marker "Function ${_q3x_symbol}:")
string(FIND "${_q3x_resources}" "${_q3x_resource_marker}"
  _q3x_resource_start)
if(_q3x_resource_start EQUAL -1)
  message(FATAL_ERROR "paired M128N512 resource record is missing")
endif()
string(SUBSTRING "${_q3x_resources}" ${_q3x_resource_start} -1
  _q3x_resource_tail)
string(REGEX MATCH
  "REG:[0-9]+ STACK:[0-9]+ SHARED:[0-9]+ LOCAL:[0-9]+"
  _q3x_record "${_q3x_resource_tail}")
if(NOT _q3x_record)
  message(FATAL_ERROR "paired M128N512 resource values are missing")
endif()
string(REGEX REPLACE ".*REG:([0-9]+).*" "\\1" _q3x_registers
  "${_q3x_record}")
string(REGEX REPLACE ".*STACK:([0-9]+).*" "\\1" _q3x_stack
  "${_q3x_record}")
string(REGEX REPLACE ".*SHARED:([0-9]+).*" "\\1" _q3x_shared
  "${_q3x_record}")
string(REGEX REPLACE ".*LOCAL:([0-9]+).*" "\\1" _q3x_local
  "${_q3x_record}")
if(_q3x_registers LESS 1 OR _q3x_registers GREATER 128 OR
   NOT _q3x_stack EQUAL 0 OR NOT _q3x_shared EQUAL 0 OR
   NOT _q3x_local EQUAL 0)
  message(FATAL_ERROR
    "paired M128N512 resource contract failed: ${_q3x_record}")
endif()

message(STATUS
  "Verified paired M128N512 feed: x4/LDS.128/IMMA=32/8/64, "
  "BAR sites=7 (dynamic K5120 N64: 1+10+9+1=21), LDGSTS=20, "
  "LDL/STL=0, ${_q3x_record}")
