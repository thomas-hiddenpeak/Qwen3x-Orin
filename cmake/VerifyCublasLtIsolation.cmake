cmake_minimum_required(VERSION 3.24)

foreach(required IN ITEMS
    Q3X_BUILD_DIR
    Q3X_AUDIT_PREFIX
    Q3X_PRODUCTION_EXECUTABLE
    Q3X_KERNEL_ARCHIVE
    Q3X_ENGINE_ARCHIVE
    Q3X_NATIVE_RETENTION_EXECUTABLE
    Q3X_INSTALLED_EXECUTABLE
    Q3X_INSTALLED_KERNEL_ARCHIVE
    Q3X_INSTALLED_ENGINE_ARCHIVE
    Q3X_NM
    Q3X_READELF)
  if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
    message(FATAL_ERROR "Missing required audit input: ${required}")
  endif()
endforeach()

file(REMOVE_RECURSE "${Q3X_AUDIT_PREFIX}")
# Seed the exact legacy public header path. An upgrade install must remove it,
# while a fresh install must never recreate it.
set(legacy_header
    "${Q3X_AUDIT_PREFIX}/include/q3x/kernels/sm87_nvfp4_prefill_cublaslt.h")
get_filename_component(legacy_header_directory "${legacy_header}" DIRECTORY)
file(MAKE_DIRECTORY "${legacy_header_directory}")
file(WRITE "${legacy_header}" "legacy header must be removed\n")
execute_process(
  COMMAND "${CMAKE_COMMAND}" --install "${Q3X_BUILD_DIR}"
          --prefix "${Q3X_AUDIT_PREFIX}"
  RESULT_VARIABLE install_status
  OUTPUT_VARIABLE install_output
  ERROR_VARIABLE install_error
)
if(NOT install_status EQUAL 0)
  message(FATAL_ERROR
    "Fresh-prefix install failed (${install_status})\n"
    "${install_output}\n${install_error}")
endif()

file(GLOB_RECURSE forbidden_installed_paths
  LIST_DIRECTORIES false
  "${Q3X_AUDIT_PREFIX}/*cublaslt*"
  "${Q3X_AUDIT_PREFIX}/*cublasLt*"
  "${Q3X_AUDIT_PREFIX}/*CUBLASLT*"
)
if(forbidden_installed_paths)
  message(FATAL_ERROR
    "Fresh install exposes forbidden cuBLASLt paths: "
    "${forbidden_installed_paths}")
endif()

function(q3x_require_no_cublaslt_symbols artifact)
  if(NOT EXISTS "${artifact}")
    message(FATAL_ERROR "Audit artifact does not exist: ${artifact}")
  endif()
  execute_process(
    COMMAND "${Q3X_NM}" -C "${artifact}"
    RESULT_VARIABLE nm_status
    OUTPUT_VARIABLE nm_output
    ERROR_VARIABLE nm_error
  )
  if(NOT nm_status EQUAL 0)
    message(FATAL_ERROR "nm failed for ${artifact}: ${nm_error}")
  endif()
  string(TOLOWER "${nm_output}" nm_output_lower)
  if(nm_output_lower MATCHES "cublaslt")
    message(FATAL_ERROR "Forbidden cuBLASLt symbol in ${artifact}")
  endif()
endfunction()

function(q3x_require_no_cublaslt_dependency artifact)
  if(NOT EXISTS "${artifact}")
    message(FATAL_ERROR "Audit artifact does not exist: ${artifact}")
  endif()
  execute_process(
    COMMAND "${Q3X_READELF}" -d "${artifact}"
    RESULT_VARIABLE readelf_status
    OUTPUT_VARIABLE readelf_output
    ERROR_VARIABLE readelf_error
  )
  if(NOT readelf_status EQUAL 0)
    message(FATAL_ERROR "readelf failed for ${artifact}: ${readelf_error}")
  endif()
  string(TOLOWER "${readelf_output}" readelf_output_lower)
  if(readelf_output_lower MATCHES "cublaslt")
    message(FATAL_ERROR "Forbidden cuBLASLt dependency in ${artifact}")
  endif()
endfunction()

foreach(production_artifact IN ITEMS
    "${Q3X_PRODUCTION_EXECUTABLE}"
    "${Q3X_KERNEL_ARCHIVE}"
    "${Q3X_ENGINE_ARCHIVE}"
    "${Q3X_INSTALLED_EXECUTABLE}"
    "${Q3X_INSTALLED_KERNEL_ARCHIVE}"
    "${Q3X_INSTALLED_ENGINE_ARCHIVE}")
  q3x_require_no_cublaslt_symbols("${production_artifact}")
endforeach()
q3x_require_no_cublaslt_dependency("${Q3X_PRODUCTION_EXECUTABLE}")
q3x_require_no_cublaslt_dependency("${Q3X_INSTALLED_EXECUTABLE}")

# The native-retention executable shares a historical source filename with the
# external-reference harness, so CUDA-local symbol names may contain that file
# stem.  Its dynamic dependency table and undefined dynamic imports must still
# be completely independent of the library.
q3x_require_no_cublaslt_dependency("${Q3X_NATIVE_RETENTION_EXECUTABLE}")
execute_process(
  COMMAND "${Q3X_NM}" -D -u -C "${Q3X_NATIVE_RETENTION_EXECUTABLE}"
  RESULT_VARIABLE native_nm_status
  OUTPUT_VARIABLE native_nm_output
  ERROR_VARIABLE native_nm_error
)
if(NOT native_nm_status EQUAL 0)
  message(FATAL_ERROR
    "dynamic nm failed for native retention executable: ${native_nm_error}")
endif()
string(TOLOWER "${native_nm_output}" native_nm_output_lower)
if(native_nm_output_lower MATCHES "cublas")
  message(FATAL_ERROR
    "Native retention executable imports a forbidden cuBLAS symbol")
endif()

message(STATUS
  "cuBLASLt isolation verified for production, fresh install, and native retention")
