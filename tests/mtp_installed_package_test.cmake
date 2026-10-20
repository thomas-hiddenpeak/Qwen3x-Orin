foreach(required IN ITEMS Q3X_BINARY_DIR Q3X_SOURCE_DIR Q3X_PYTHON Q3X_MTP_CAP)
  if(NOT DEFINED ${required})
    message(FATAL_ERROR "Missing ${required}")
  endif()
endforeach()
file(REAL_PATH "${Q3X_BINARY_DIR}" binary_dir)
file(REAL_PATH "${Q3X_SOURCE_DIR}/.q3x-work" work_dir)
cmake_path(IS_PREFIX work_dir "${binary_dir}" NORMALIZE inside_workspace)
if(NOT inside_workspace OR NOT Q3X_MTP_CAP MATCHES "^[23]$")
  message(FATAL_ERROR "MTP package test requires a workspace build and cap 2/3")
endif()
set(prefix "${binary_dir}/mtp-package-test-install")
file(REMOVE_RECURSE "${prefix}")
execute_process(COMMAND "${CMAKE_COMMAND}" --install "${binary_dir}" --prefix "${prefix}"
  COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND "${Q3X_PYTHON}" -B
  "${prefix}/share/Qwen3xOrinMTP/tools/audit_mtp_install.py"
  --prefix "${prefix}" --cap "${Q3X_MTP_CAP}"
  COMMAND_ERROR_IS_FATAL ANY)
