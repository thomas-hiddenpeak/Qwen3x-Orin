if(NOT DEFINED Q3X_PTX_FILE OR NOT EXISTS "${Q3X_PTX_FILE}")
  message(FATAL_ERROR "SM87 A4W4 Gate+Up PTX sentinel file is missing")
endif()

file(READ "${Q3X_PTX_FILE}" _q3x_gateup_ptx)

foreach(_q3x_required_instruction IN ITEMS
    "mma[.]sync[.]aligned[.]m16n8k64[.]row[.]col[.]s32[.]s4[.]s4[.]s32"
    "cp[.]async[.]cg[.]shared[.]global"
    "cp[.]async[.]commit_group"
    "cp[.]async[.]wait_group")
  string(REGEX MATCH "${_q3x_required_instruction}" _q3x_match
                     "${_q3x_gateup_ptx}")
  if(NOT _q3x_match)
    message(FATAL_ERROR
      "SM87 A4W4 Gate+Up PTX is missing ${_q3x_required_instruction}"
    )
  endif()
endforeach()

string(REGEX MATCH "[.]target[ \t]+sm_87" _q3x_sm87_target_match
                   "${_q3x_gateup_ptx}")
if(NOT _q3x_sm87_target_match)
  message(FATAL_ERROR "SM87 A4W4 Gate+Up PTX is not targeted at sm_87")
endif()

message(STATUS
  "Verified paired Gate+Up SM87 S4 MMA and cp.async PTX contract"
)
