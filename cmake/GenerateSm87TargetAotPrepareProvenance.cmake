if(NOT DEFINED Q3X_PROVENANCE_SOURCE_DIR OR
   NOT DEFINED Q3X_PROVENANCE_GIT_EXECUTABLE OR
   NOT DEFINED Q3X_PROVENANCE_TEMPLATE OR
   NOT DEFINED Q3X_PROVENANCE_OUTPUT)
  message(FATAL_ERROR "incomplete target-AOT prepare provenance inputs")
endif()

execute_process(
  COMMAND "${Q3X_PROVENANCE_GIT_EXECUTABLE}" -C
          "${Q3X_PROVENANCE_SOURCE_DIR}" rev-parse HEAD
  OUTPUT_VARIABLE Q3X_PROVENANCE_GIT_COMMIT
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE git_commit_status
  ERROR_VARIABLE git_commit_error
)
execute_process(
  COMMAND "${Q3X_PROVENANCE_GIT_EXECUTABLE}" -C
          "${Q3X_PROVENANCE_SOURCE_DIR}" rev-parse "HEAD^{tree}"
  OUTPUT_VARIABLE Q3X_PROVENANCE_GIT_TREE
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE git_tree_status
  ERROR_VARIABLE git_tree_error
)
execute_process(
  COMMAND "${Q3X_PROVENANCE_GIT_EXECUTABLE}" -C
          "${Q3X_PROVENANCE_SOURCE_DIR}" status --porcelain=v1
          --untracked-files=all
  OUTPUT_VARIABLE git_status_porcelain
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE git_status_status
  ERROR_VARIABLE git_status_error
)
if(NOT git_commit_status EQUAL 0 OR NOT git_tree_status EQUAL 0 OR
   NOT git_status_status EQUAL 0)
  message(FATAL_ERROR
    "cannot freeze target-AOT prepare git provenance: "
    "${git_commit_error}${git_tree_error}${git_status_error}"
  )
endif()

if(git_status_porcelain STREQUAL "")
  set(Q3X_PROVENANCE_GIT_CLEAN true)
else()
  set(Q3X_PROVENANCE_GIT_CLEAN false)
endif()

if(EXISTS "${Q3X_PROVENANCE_CXX_COMPILER}")
  file(SHA256 "${Q3X_PROVENANCE_CXX_COMPILER}"
       Q3X_PROVENANCE_CXX_COMPILER_SHA256)
else()
  set(Q3X_PROVENANCE_CXX_COMPILER_SHA256 "unavailable")
endif()
if(EXISTS "${Q3X_PROVENANCE_CUDA_COMPILER}")
  file(SHA256 "${Q3X_PROVENANCE_CUDA_COMPILER}"
       Q3X_PROVENANCE_CUDA_COMPILER_SHA256)
else()
  set(Q3X_PROVENANCE_CUDA_COMPILER_SHA256 "unavailable")
endif()

foreach(boolean_name IN ITEMS
    Q3X_PROVENANCE_BUILD_TESTING
    Q3X_PROVENANCE_AOT_SYSTEM_ADMISSION
    Q3X_PROVENANCE_TARGET_PROJECTION_ADMISSION
    Q3X_PROVENANCE_TARGET_DEVICE_ASSETS_ADMISSION
    Q3X_PROVENANCE_TARGET_LAYER0_M192_ORACLE_ADMISSION
    Q3X_PROVENANCE_FP8_MARLIN_PREFILL_ADMISSION
    Q3X_PROVENANCE_NVFP4_MARLIN_PREFILL_ADMISSION
    Q3X_PROVENANCE_P40_PACKED_PROJECTION_ADMISSION
    Q3X_PROVENANCE_P40_PACKED_NVFP4_V2_ADMISSION
    Q3X_PROVENANCE_P40_VLLM_MARLIN_PARITY_ADMISSION)
  if(${boolean_name})
    set(${boolean_name} true)
  else()
    set(${boolean_name} false)
  endif()
endforeach()

string(CONCAT build_receipt
  "schema=q3x.sm87.target-aot.prepare-build.v1\n"
  "git_commit=${Q3X_PROVENANCE_GIT_COMMIT}\n"
  "git_tree=${Q3X_PROVENANCE_GIT_TREE}\n"
  "git_clean=${Q3X_PROVENANCE_GIT_CLEAN}\n"
  "cmake_version=${Q3X_PROVENANCE_CMAKE_VERSION}\n"
  "generator=${Q3X_PROVENANCE_GENERATOR}\n"
  "build_type=${Q3X_PROVENANCE_BUILD_TYPE}\n"
  "build_testing=${Q3X_PROVENANCE_BUILD_TESTING}\n"
  "q3x_cuda_architectures=${Q3X_PROVENANCE_Q3X_CUDA_ARCHITECTURES}\n"
  "effective_cuda_architectures=${Q3X_PROVENANCE_EFFECTIVE_CUDA_ARCHITECTURES}\n"
  "cxx_compiler=${Q3X_PROVENANCE_CXX_COMPILER}\n"
  "cxx_compiler_id=${Q3X_PROVENANCE_CXX_COMPILER_ID}\n"
  "cxx_compiler_version=${Q3X_PROVENANCE_CXX_COMPILER_VERSION}\n"
  "cxx_compiler_sha256=${Q3X_PROVENANCE_CXX_COMPILER_SHA256}\n"
  "cuda_compiler=${Q3X_PROVENANCE_CUDA_COMPILER}\n"
  "cuda_compiler_id=${Q3X_PROVENANCE_CUDA_COMPILER_ID}\n"
  "cuda_compiler_version=${Q3X_PROVENANCE_CUDA_COMPILER_VERSION}\n"
  "cuda_compiler_sha256=${Q3X_PROVENANCE_CUDA_COMPILER_SHA256}\n"
  "cuda_toolkit_version=${Q3X_PROVENANCE_CUDA_TOOLKIT_VERSION}\n"
  "cxx_release_flags=${Q3X_PROVENANCE_CXX_RELEASE_FLAGS}\n"
  "cuda_release_flags=${Q3X_PROVENANCE_CUDA_RELEASE_FLAGS}\n"
  "cxx_global_flags=${Q3X_PROVENANCE_CXX_GLOBAL_FLAGS}\n"
  "cuda_global_flags=${Q3X_PROVENANCE_CUDA_GLOBAL_FLAGS}\n"
  "aot_system_admission=${Q3X_PROVENANCE_AOT_SYSTEM_ADMISSION}\n"
  "target_projection_admission=${Q3X_PROVENANCE_TARGET_PROJECTION_ADMISSION}\n"
  "target_device_assets_admission=${Q3X_PROVENANCE_TARGET_DEVICE_ASSETS_ADMISSION}\n"
  "target_layer0_m192_oracle_admission=${Q3X_PROVENANCE_TARGET_LAYER0_M192_ORACLE_ADMISSION}\n"
  "fp8_marlin_prefill_admission=${Q3X_PROVENANCE_FP8_MARLIN_PREFILL_ADMISSION}\n"
  "nvfp4_marlin_prefill_admission=${Q3X_PROVENANCE_NVFP4_MARLIN_PREFILL_ADMISSION}\n"
  "p40_packed_projection_admission=${Q3X_PROVENANCE_P40_PACKED_PROJECTION_ADMISSION}\n"
  "p40_packed_nvfp4_v2_admission=${Q3X_PROVENANCE_P40_PACKED_NVFP4_V2_ADMISSION}\n"
  "p40_vllm_marlin_parity_admission=${Q3X_PROVENANCE_P40_VLLM_MARLIN_PARITY_ADMISSION}\n"
)
string(SHA256 Q3X_PROVENANCE_BUILD_RECEIPT_SHA256 "${build_receipt}")

foreach(value_name IN ITEMS
    Q3X_PROVENANCE_GIT_COMMIT
    Q3X_PROVENANCE_GIT_TREE
    Q3X_PROVENANCE_CMAKE_VERSION
    Q3X_PROVENANCE_GENERATOR
    Q3X_PROVENANCE_BUILD_TYPE
    Q3X_PROVENANCE_Q3X_CUDA_ARCHITECTURES
    Q3X_PROVENANCE_EFFECTIVE_CUDA_ARCHITECTURES
    Q3X_PROVENANCE_CXX_COMPILER
    Q3X_PROVENANCE_CXX_COMPILER_ID
    Q3X_PROVENANCE_CXX_COMPILER_VERSION
    Q3X_PROVENANCE_CXX_COMPILER_SHA256
    Q3X_PROVENANCE_CUDA_COMPILER
    Q3X_PROVENANCE_CUDA_COMPILER_ID
    Q3X_PROVENANCE_CUDA_COMPILER_VERSION
    Q3X_PROVENANCE_CUDA_COMPILER_SHA256
    Q3X_PROVENANCE_CUDA_TOOLKIT_VERSION
    Q3X_PROVENANCE_CXX_RELEASE_FLAGS
    Q3X_PROVENANCE_CUDA_RELEASE_FLAGS
    Q3X_PROVENANCE_CXX_GLOBAL_FLAGS
    Q3X_PROVENANCE_CUDA_GLOBAL_FLAGS
    Q3X_PROVENANCE_BUILD_RECEIPT_SHA256)
  string(REPLACE "\\" "\\\\" ${value_name} "${${value_name}}")
  string(REPLACE "\"" "\\\"" ${value_name} "${${value_name}}")
  string(REPLACE "\n" "\\n" ${value_name} "${${value_name}}")
endforeach()

get_filename_component(output_directory "${Q3X_PROVENANCE_OUTPUT}" DIRECTORY)
file(MAKE_DIRECTORY "${output_directory}")
set(temporary_output "${Q3X_PROVENANCE_OUTPUT}.tmp")
configure_file("${Q3X_PROVENANCE_TEMPLATE}" "${temporary_output}" @ONLY)
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different
          "${temporary_output}" "${Q3X_PROVENANCE_OUTPUT}"
  RESULT_VARIABLE copy_status
)
file(REMOVE "${temporary_output}")
if(NOT copy_status EQUAL 0)
  message(FATAL_ERROR "cannot publish target-AOT prepare provenance header")
endif()
