if(NOT DEFINED Q3X_CUOBJDUMP_EXECUTABLE OR
   NOT EXISTS "${Q3X_CUOBJDUMP_EXECUTABLE}")
  message(FATAL_ERROR "cuobjdump is unavailable for the LDSM operand probe")
endif()
if(NOT DEFINED Q3X_LDMATRIX_OPERAND_PROBE_BINARY OR
   NOT EXISTS "${Q3X_LDMATRIX_OPERAND_PROBE_BINARY}")
  message(FATAL_ERROR "LDSM operand probe binary is missing")
endif()
if(NOT DEFINED Q3X_LDMATRIX_OPERAND_PROBE_AUDIT_DIR)
  set(Q3X_LDMATRIX_OPERAND_PROBE_AUDIT_DIR
      "${CMAKE_CURRENT_BINARY_DIR}/sm87_a4w4_ldmatrix_operand_probe_audit")
endif()
file(MAKE_DIRECTORY "${Q3X_LDMATRIX_OPERAND_PROBE_AUDIT_DIR}")

execute_process(
  COMMAND "${Q3X_CUOBJDUMP_EXECUTABLE}" --dump-sass
          "${Q3X_LDMATRIX_OPERAND_PROBE_BINARY}"
  RESULT_VARIABLE _q3x_sass_result
  OUTPUT_VARIABLE _q3x_sass
  ERROR_VARIABLE _q3x_sass_error
)
if(NOT _q3x_sass_result EQUAL 0)
  message(FATAL_ERROR "LDSM operand-probe SASS dump failed: ${_q3x_sass_error}")
endif()
file(WRITE
  "${Q3X_LDMATRIX_OPERAND_PROBE_AUDIT_DIR}/operand-probe.sass"
  "${_q3x_sass}"
)

function(q3x_extract_sass_function input symbol output)
  set(_q3x_marker "Function : ${symbol}")
  string(FIND "${input}" "${_q3x_marker}" _q3x_start)
  if(_q3x_start EQUAL -1)
    message(FATAL_ERROR "SASS symbol is missing: ${symbol}")
  endif()
  string(SUBSTRING "${input}" ${_q3x_start} -1 _q3x_tail)
  string(FIND "${_q3x_tail}" "\n\t\tFunction : " _q3x_next)
  if(NOT _q3x_next EQUAL -1)
    string(SUBSTRING "${_q3x_tail}" 0 ${_q3x_next} _q3x_tail)
  endif()
  set(${output} "${_q3x_tail}" PARENT_SCOPE)
endfunction()

q3x_extract_sass_function(
  "${_q3x_sass}"
  "q3x_sm87_a4w4_ldmatrix_operand_probe_kernel"
  _q3x_ldmatrix_kernel_sass
)
q3x_extract_sass_function(
  "${_q3x_sass}"
  "q3x_sm87_a4w4_scalar_lds_operand_probe_kernel"
  _q3x_scalar_kernel_sass
)
q3x_extract_sass_function(
  "${_q3x_sass}"
  "q3x_sm87_a4w4_scalar_a_ldmatrix_b_operand_probe_kernel"
  _q3x_scalar_a_ldmatrix_b_kernel_sass
)
q3x_extract_sass_function(
  "${_q3x_sass}"
  "q3x_sm87_a4w4_ldmatrix_a_scalar_b_operand_probe_kernel"
  _q3x_ldmatrix_a_scalar_b_kernel_sass
)
file(WRITE
  "${Q3X_LDMATRIX_OPERAND_PROBE_AUDIT_DIR}/ldmatrix-kernel.sass"
  "${_q3x_ldmatrix_kernel_sass}"
)
file(WRITE
  "${Q3X_LDMATRIX_OPERAND_PROBE_AUDIT_DIR}/scalar-kernel.sass"
  "${_q3x_scalar_kernel_sass}"
)
file(WRITE
  "${Q3X_LDMATRIX_OPERAND_PROBE_AUDIT_DIR}/scalar-a-ldmatrix-b-kernel.sass"
  "${_q3x_scalar_a_ldmatrix_b_kernel_sass}"
)
file(WRITE
  "${Q3X_LDMATRIX_OPERAND_PROBE_AUDIT_DIR}/ldmatrix-a-scalar-b-kernel.sass"
  "${_q3x_ldmatrix_a_scalar_b_kernel_sass}"
)

function(q3x_verify_operand_contract sass label expected_lds
         expected_x4 expected_x2)
  string(REGEX MATCHALL "[ 	]LDS[ 	]" _q3x_lds "${sass}")
  string(REGEX MATCHALL "LDSM[.]16[.]M88[.]4[ 	]"
    _q3x_x4 "${sass}")
  string(REGEX MATCHALL "LDSM[.]16[.]M88[.]2[ 	]"
    _q3x_x2 "${sass}")
  string(REGEX MATCHALL "LDSM[.]16[.]M88[.]1[ 	]"
    _q3x_x1 "${sass}")
  string(REGEX MATCHALL "IMMA[.]16864[.]S4[.]S4[ 	]"
    _q3x_imma "${sass}")
  list(LENGTH _q3x_lds _q3x_lds_count)
  list(LENGTH _q3x_x4 _q3x_x4_count)
  list(LENGTH _q3x_x2 _q3x_x2_count)
  list(LENGTH _q3x_x1 _q3x_x1_count)
  list(LENGTH _q3x_imma _q3x_imma_count)
  if(NOT _q3x_lds_count EQUAL expected_lds OR
     NOT _q3x_x4_count EQUAL expected_x4 OR
     NOT _q3x_x2_count EQUAL expected_x2 OR
     NOT _q3x_x1_count EQUAL 0 OR
     NOT _q3x_imma_count EQUAL 1)
    message(FATAL_ERROR
      "${label} operand contract failed: LDS=${_q3x_lds_count} "
      "(expected ${expected_lds}), x4=${_q3x_x4_count} "
      "(expected ${expected_x4}), x2=${_q3x_x2_count} "
      "(expected ${expected_x2}), x1=${_q3x_x1_count} (expected 0), "
      "IMMA=${_q3x_imma_count} (expected 1)"
    )
  endif()
  foreach(_q3x_forbidden IN ITEMS PRMT SHFL LDL STL)
    string(REGEX MATCHALL
      "[ 	]${_q3x_forbidden}([.][A-Za-z0-9]+)*[ 	]"
      _q3x_forbidden_matches "${sass}"
    )
    list(LENGTH _q3x_forbidden_matches _q3x_forbidden_count)
    if(NOT _q3x_forbidden_count EQUAL 0)
      message(FATAL_ERROR
        "${label} emitted ${_q3x_forbidden_count} "
        "${_q3x_forbidden} instruction(s)"
      )
    endif()
  endforeach()
endfunction()

q3x_verify_operand_contract("${_q3x_scalar_kernel_sass}"
  "scalar-A/scalar-B" 6 0 0)
q3x_verify_operand_contract("${_q3x_ldmatrix_kernel_sass}"
  "LDSM-A/LDSM-B" 0 1 1)
q3x_verify_operand_contract("${_q3x_scalar_a_ldmatrix_b_kernel_sass}"
  "scalar-A/LDSM-B" 4 0 1)
q3x_verify_operand_contract("${_q3x_ldmatrix_a_scalar_b_kernel_sass}"
  "LDSM-A/scalar-B" 2 1 0)

execute_process(
  COMMAND "${Q3X_CUOBJDUMP_EXECUTABLE}" --dump-resource-usage
          "${Q3X_LDMATRIX_OPERAND_PROBE_BINARY}"
  RESULT_VARIABLE _q3x_resource_result
  OUTPUT_VARIABLE _q3x_resources
  ERROR_VARIABLE _q3x_resource_error
)
if(NOT _q3x_resource_result EQUAL 0)
  message(FATAL_ERROR
    "LDSM operand-probe resource dump failed: ${_q3x_resource_error}"
  )
endif()
file(WRITE
  "${Q3X_LDMATRIX_OPERAND_PROBE_AUDIT_DIR}/operand-probe.resources"
  "${_q3x_resources}"
)

function(q3x_verify_resource_contract resources symbol label output_regs)
  set(_q3x_marker "Function ${symbol}:")
  string(FIND "${resources}" "${_q3x_marker}" _q3x_start)
  if(_q3x_start EQUAL -1)
    message(FATAL_ERROR "${label} resource record is missing")
  endif()
  string(SUBSTRING "${resources}" ${_q3x_start} -1 _q3x_tail)
  string(REGEX MATCH
    "REG:[0-9]+ STACK:[0-9]+ SHARED:[0-9]+ LOCAL:[0-9]+"
    _q3x_record "${_q3x_tail}"
  )
  if(NOT _q3x_record)
    message(FATAL_ERROR "${label} resource values are missing")
  endif()
  string(REGEX REPLACE ".*REG:([0-9]+).*" "\\1"
    _q3x_registers "${_q3x_record}")
  string(REGEX REPLACE ".*STACK:([0-9]+).*" "\\1"
    _q3x_stack "${_q3x_record}")
  string(REGEX REPLACE ".*SHARED:([0-9]+).*" "\\1"
    _q3x_shared "${_q3x_record}")
  string(REGEX REPLACE ".*LOCAL:([0-9]+).*" "\\1"
    _q3x_local "${_q3x_record}")
  if(NOT _q3x_stack EQUAL 0 OR NOT _q3x_local EQUAL 0 OR
     NOT _q3x_shared EQUAL 768)
    message(FATAL_ERROR "${label} resource contract failed: ${_q3x_record}")
  endif()
  set(${output_regs} "${_q3x_registers}" PARENT_SCOPE)
endfunction()

q3x_verify_resource_contract("${_q3x_resources}"
  "q3x_sm87_a4w4_scalar_lds_operand_probe_kernel"
  "scalar-A/scalar-B" _q3x_scalar_registers)
q3x_verify_resource_contract("${_q3x_resources}"
  "q3x_sm87_a4w4_ldmatrix_operand_probe_kernel"
  "LDSM-A/LDSM-B" _q3x_ldmatrix_registers)
q3x_verify_resource_contract("${_q3x_resources}"
  "q3x_sm87_a4w4_scalar_a_ldmatrix_b_operand_probe_kernel"
  "scalar-A/LDSM-B" _q3x_scalar_a_ldmatrix_b_registers)
q3x_verify_resource_contract("${_q3x_resources}"
  "q3x_sm87_a4w4_ldmatrix_a_scalar_b_operand_probe_kernel"
  "LDSM-A/scalar-B" _q3x_ldmatrix_a_scalar_b_registers)

message(STATUS
  "Verified four operand probes: scalar=6 LDS; full=0 LDS+x4+x2; "
  "scalar-A/LDSM-B=4 LDS+x2; LDSM-A/scalar-B=2 LDS+x4; "
  "each has 1 IMMA, PRMT/SHFL/LDL/STL=0, shared=768, "
  "regs=${_q3x_scalar_registers}/${_q3x_ldmatrix_registers}/"
  "${_q3x_scalar_a_ldmatrix_b_registers}/"
  "${_q3x_ldmatrix_a_scalar_b_registers}"
)
