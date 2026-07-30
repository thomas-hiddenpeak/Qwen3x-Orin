if(NOT DEFINED Q3X_PTX_FILE OR NOT EXISTS "${Q3X_PTX_FILE}")
  message(FATAL_ERROR "SM87 A4W4 PTX sentinel file is missing")
endif()

file(READ "${Q3X_PTX_FILE}" _q3x_a4w4_ptx)
string(REGEX MATCH
  "mma[.]sync[.]aligned[.]m16n8k64[.]row[.]col[.]s32[.]s4[.]s4[.]s32"
  _q3x_a4w4_mma_match
  "${_q3x_a4w4_ptx}"
)
if(NOT _q3x_a4w4_mma_match)
  message(FATAL_ERROR
    "SM87 A4W4 PTX does not contain m16n8k64.row.col.s32.s4.s4.s32"
  )
endif()

string(REGEX MATCH "[.]target[ \t]+sm_87" _q3x_sm87_target_match
                   "${_q3x_a4w4_ptx}")
if(NOT _q3x_sm87_target_match)
  message(FATAL_ERROR "SM87 A4W4 PTX sentinel is not targeted at sm_87")
endif()

message(STATUS
  "Verified native SM87 A4W4 PTX: ${_q3x_a4w4_mma_match}"
)
