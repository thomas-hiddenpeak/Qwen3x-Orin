if(NOT DEFINED Q3X_CUOBJDUMP_EXECUTABLE OR
   NOT EXISTS "${Q3X_CUOBJDUMP_EXECUTABLE}")
  message(FATAL_ERROR "cuobjdump is unavailable for the Down LDSM pair-ring audit")
endif()
if(NOT DEFINED Q3X_DOWN_LDMATRIX_PAIRRING_BINARY OR
   NOT EXISTS "${Q3X_DOWN_LDMATRIX_PAIRRING_BINARY}")
  message(FATAL_ERROR "Down LDSM pair-ring binary is missing")
endif()
if(NOT DEFINED Q3X_DOWN_LDMATRIX_PAIRRING_AUDIT_DIR)
  set(Q3X_DOWN_LDMATRIX_PAIRRING_AUDIT_DIR
      "${CMAKE_CURRENT_BINARY_DIR}/sm87_a4w4_down_k512_m128n128_ldmatrix_pairring_audit")
endif()
file(MAKE_DIRECTORY "${Q3X_DOWN_LDMATRIX_PAIRRING_AUDIT_DIR}")

execute_process(
  COMMAND "${Q3X_CUOBJDUMP_EXECUTABLE}" --dump-sass
          "${Q3X_DOWN_LDMATRIX_PAIRRING_BINARY}"
  RESULT_VARIABLE _q3x_sass_result
  OUTPUT_VARIABLE _q3x_sass
  ERROR_VARIABLE _q3x_sass_error
)
if(NOT _q3x_sass_result EQUAL 0)
  message(FATAL_ERROR "Down LDSM pair-ring SASS dump failed: ${_q3x_sass_error}")
endif()
file(WRITE "${Q3X_DOWN_LDMATRIX_PAIRRING_AUDIT_DIR}/binary.sass"
  "${_q3x_sass}")

set(_q3x_symbol
  "q3x_sm87_a4w4_down_k512_m128n128_ldmatrix_pairring_kernel")
set(_q3x_marker "Function : ${_q3x_symbol}")
string(FIND "${_q3x_sass}" "${_q3x_marker}" _q3x_start)
if(_q3x_start EQUAL -1)
  message(FATAL_ERROR "Down LDSM pair-ring SASS symbol is missing")
endif()
string(SUBSTRING "${_q3x_sass}" ${_q3x_start} -1 _q3x_kernel)
string(FIND "${_q3x_kernel}" "\n\t\tFunction : " _q3x_next)
if(NOT _q3x_next EQUAL -1)
  string(SUBSTRING "${_q3x_kernel}" 0 ${_q3x_next} _q3x_kernel)
endif()
file(WRITE "${Q3X_DOWN_LDMATRIX_PAIRRING_AUDIT_DIR}/kernel.sass"
  "${_q3x_kernel}")

function(q3x_count_matches input regex output)
  string(REGEX MATCHALL "${regex}" _q3x_matches "${input}")
  list(LENGTH _q3x_matches _q3x_count)
  set(${output} "${_q3x_count}" PARENT_SCOPE)
endfunction()

q3x_count_matches("${_q3x_kernel}" "LDSM[.]16[.]M88[.]4[ \t]"
  _q3x_ldsm_x4)
q3x_count_matches("${_q3x_kernel}" "LDSM[.]16[.]M88[.]2[ \t]"
  _q3x_ldsm_x2)
q3x_count_matches("${_q3x_kernel}" "LDSM[.]16[.]M88[.]1[ \t]"
  _q3x_ldsm_x1)
q3x_count_matches("${_q3x_kernel}" "IMMA[.]16864[.]S4[.]S4[ \t]"
  _q3x_imma)
q3x_count_matches("${_q3x_kernel}" "[ \t]BAR[.]SYNC([.][A-Z_]+)*[ \t]"
  _q3x_bar_sites)
q3x_count_matches("${_q3x_kernel}" "[ \t]LDGSTS([.][A-Z0-9_]+)*[ \t]"
  _q3x_ldgsts)
q3x_count_matches("${_q3x_kernel}" "[ \t]LDS[ \t]"
  _q3x_scalar_lds)

if(NOT _q3x_ldsm_x4 EQUAL 16 OR NOT _q3x_ldsm_x2 EQUAL 64 OR
   NOT _q3x_ldsm_x1 EQUAL 0 OR NOT _q3x_imma EQUAL 128 OR
   NOT _q3x_bar_sites EQUAL 3 OR NOT _q3x_ldgsts EQUAL 32 OR
   NOT _q3x_scalar_lds EQUAL 6)
  message(FATAL_ERROR
    "Down LDSM pair-ring instruction contract failed: "
    "x4=${_q3x_ldsm_x4}/16 x2=${_q3x_ldsm_x2}/64 "
    "x1=${_q3x_ldsm_x1}/0 IMMA=${_q3x_imma}/128 "
    "BAR-sites=${_q3x_bar_sites}/3 LDGSTS=${_q3x_ldgsts}/32 "
    "scalar-LDS=${_q3x_scalar_lds}/6"
  )
endif()

# ptxas emits six false-predicated RZ->RZ LDS placeholders next to the two
# cp.async issue regions.  They can never execute and carry no operand data.
# Strip exactly that inert line form, then reject any remaining scalar LDS.
string(REGEX REPLACE
  "[^\n]*@!PT[ \t]+LDS[ \t]+RZ,[ \t]*\\[RZ\\][^\n]*\n"
  "" _q3x_without_inert_lds "${_q3x_kernel}")
q3x_count_matches("${_q3x_without_inert_lds}" "[ \t]LDS[ \t]"
  _q3x_executable_scalar_lds)
if(NOT _q3x_executable_scalar_lds EQUAL 0)
  message(FATAL_ERROR
    "Down LDSM pair-ring emitted ${_q3x_executable_scalar_lds} "
    "executable scalar LDS instruction(s)")
endif()

foreach(_q3x_forbidden IN ITEMS LDL STL PRMT SHFL)
  q3x_count_matches("${_q3x_kernel}"
    "[ \t]${_q3x_forbidden}([.][A-Za-z0-9_]+)*[ \t]"
    _q3x_forbidden_count)
  if(NOT _q3x_forbidden_count EQUAL 0)
    message(FATAL_ERROR
      "Down LDSM pair-ring emitted ${_q3x_forbidden_count} "
      "${_q3x_forbidden} instruction(s)")
  endif()
endforeach()

execute_process(
  COMMAND "${Q3X_CUOBJDUMP_EXECUTABLE}" --dump-resource-usage
          "${Q3X_DOWN_LDMATRIX_PAIRRING_BINARY}"
  RESULT_VARIABLE _q3x_resource_result
  OUTPUT_VARIABLE _q3x_resources
  ERROR_VARIABLE _q3x_resource_error
)
if(NOT _q3x_resource_result EQUAL 0)
  message(FATAL_ERROR
    "Down LDSM pair-ring resource dump failed: ${_q3x_resource_error}")
endif()
file(WRITE "${Q3X_DOWN_LDMATRIX_PAIRRING_AUDIT_DIR}/binary.resources"
  "${_q3x_resources}")

set(_q3x_resource_marker "Function ${_q3x_symbol}:")
string(FIND "${_q3x_resources}" "${_q3x_resource_marker}"
  _q3x_resource_start)
if(_q3x_resource_start EQUAL -1)
  message(FATAL_ERROR "Down LDSM pair-ring resource record is missing")
endif()
string(SUBSTRING "${_q3x_resources}" ${_q3x_resource_start} -1
  _q3x_resource_tail)
string(REGEX MATCH
  "REG:[0-9]+ STACK:[0-9]+ SHARED:[0-9]+ LOCAL:[0-9]+"
  _q3x_record "${_q3x_resource_tail}")
if(NOT _q3x_record)
  message(FATAL_ERROR "Down LDSM pair-ring resource values are missing")
endif()
string(REGEX REPLACE ".*REG:([0-9]+).*" "\\1" _q3x_registers
  "${_q3x_record}")
string(REGEX REPLACE ".*STACK:([0-9]+).*" "\\1" _q3x_stack
  "${_q3x_record}")
string(REGEX REPLACE ".*SHARED:([0-9]+).*" "\\1" _q3x_shared
  "${_q3x_record}")
string(REGEX REPLACE ".*LOCAL:([0-9]+).*" "\\1" _q3x_local
  "${_q3x_record}")
if(_q3x_registers LESS 1 OR _q3x_registers GREATER 255 OR
   NOT _q3x_stack EQUAL 0 OR NOT _q3x_shared EQUAL 0 OR
   NOT _q3x_local EQUAL 0)
  message(FATAL_ERROR
    "Down LDSM pair-ring resource contract failed: ${_q3x_record}")
endif()

message(STATUS
  "Verified Down M128N128 LDSM pair-ring: regs=${_q3x_registers}, "
  "stack/local=0, x4/x2/IMMA=16/64/128, executable scalar LDS=0, "
  "BAR sites=3 (dynamic P1853 tile: 1+33+1=35), LDGSTS=32"
)
