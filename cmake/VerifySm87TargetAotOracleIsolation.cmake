cmake_minimum_required(VERSION 3.24)

foreach(required IN ITEMS Q3X_KERNEL_ARCHIVE Q3X_NM)
  if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
    message(FATAL_ERROR "Missing required audit input: ${required}")
  endif()
endforeach()

if(NOT EXISTS "${Q3X_KERNEL_ARCHIVE}")
  message(FATAL_ERROR
    "Target-AOT kernel archive does not exist: ${Q3X_KERNEL_ARCHIVE}")
endif()

execute_process(
  COMMAND "${Q3X_NM}" -C "${Q3X_KERNEL_ARCHIVE}"
  RESULT_VARIABLE nm_status
  OUTPUT_VARIABLE nm_output
  ERROR_VARIABLE nm_error
)
if(NOT nm_status EQUAL 0)
  message(FATAL_ERROR
    "nm failed for ${Q3X_KERNEL_ARCHIVE}: ${nm_error}")
endif()

foreach(forbidden IN ITEMS
    "sm87_target_aot_layer0_m192_oracle_detail"
    "sm87_target_aot_nvfp4_down_m192_oracle_kernel"
    "layer0_m192_oracle")
  if(nm_output MATCHES "${forbidden}")
    message(FATAL_ERROR
      "Oracle-OFF target-AOT archive exposes forbidden symbol: ${forbidden}")
  endif()
endforeach()

# Freeze the non-Oracle CUDA kernel ABI. The public launcher remains a
# fail-closed sentinel, but its compiled resource body must retain the
# original in-place Down residual signature when the M192 oracle is absent.
set(expected_down_abi
  "sm87_target_aot_nvfp4_down_kernel(unsigned short const*, unsigned char const*, unsigned int, float, unsigned short*)")
string(FIND "${nm_output}" "${expected_down_abi}" expected_down_abi_index)
if(expected_down_abi_index EQUAL -1)
  message(FATAL_ERROR
    "Oracle-OFF archive does not retain the original in-place Down kernel ABI")
endif()

set(forbidden_oracle_down_abi
  "sm87_target_aot_nvfp4_down_kernel(unsigned short const*, unsigned char const*, unsigned int, float, unsigned short const*, unsigned short*)")
string(FIND "${nm_output}" "${forbidden_oracle_down_abi}"
  forbidden_oracle_down_abi_index)
if(NOT forbidden_oracle_down_abi_index EQUAL -1)
  message(FATAL_ERROR
    "Oracle-OFF archive changed the ordinary Down kernel to the Oracle ABI")
endif()

message(STATUS
  "SM87 target-AOT M192 oracle symbols and ABI are isolated when admission is OFF")
