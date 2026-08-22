if(NOT DEFINED Q3X_SOURCE_DIR OR NOT DEFINED Q3X_BINARY_DIR OR
   NOT DEFINED Q3X_GENERATOR OR NOT DEFINED Q3X_INSTALL_BINDIR)
  message(FATAL_ERROR "package consumer test is missing required paths")
endif()

set(prefix "${Q3X_BINARY_DIR}/package-consumer-prefix")
set(consumer_binary "${Q3X_BINARY_DIR}/package-consumer-build")
file(REMOVE_RECURSE "${prefix}" "${consumer_binary}")

execute_process(
  COMMAND "${CMAKE_COMMAND}" --install "${Q3X_BINARY_DIR}"
          --prefix "${prefix}" --config "${Q3X_CONFIG}"
  RESULT_VARIABLE install_result
  OUTPUT_VARIABLE install_output
  ERROR_VARIABLE install_error
)
if(NOT install_result EQUAL 0)
  message(FATAL_ERROR
    "package install failed (${install_result})\n${install_output}${install_error}")
endif()

foreach(executable qwen3x-orin qwen3x-inspect)
  if(NOT EXISTS "${prefix}/${Q3X_INSTALL_BINDIR}/${executable}")
    message(FATAL_ERROR
      "package install is missing ${Q3X_INSTALL_BINDIR}/${executable}")
  endif()
endforeach()
if(Q3X_EXPECT_EVAL_SERVER AND
   NOT EXISTS "${prefix}/${Q3X_INSTALL_BINDIR}/qwen3x-eval-server")
  message(FATAL_ERROR
    "package install is missing ${Q3X_INSTALL_BINDIR}/qwen3x-eval-server")
endif()
if(NOT Q3X_EXPECT_EVAL_SERVER AND
   EXISTS "${prefix}/${Q3X_INSTALL_BINDIR}/qwen3x-eval-server")
  message(FATAL_ERROR
    "development package unexpectedly installed qwen3x-eval-server")
endif()
if(Q3X_EXPECT_EVAL_SERVER)
  execute_process(
    COMMAND "${prefix}/${Q3X_INSTALL_BINDIR}/qwen3x-eval-server" --help
    RESULT_VARIABLE server_help_result
    OUTPUT_VARIABLE server_help_output
    ERROR_VARIABLE server_help_error
  )
  if(NOT server_help_result EQUAL 0 OR
     NOT server_help_output MATCHES "evaluation server 0\\.7\\.0" OR
     NOT server_help_output MATCHES
       "q3x\\.sm87\\.production\\.p40\\.legacy-c512-exact\\.v3")
    message(FATAL_ERROR
      "installed production server identity mismatch\n"
      "${server_help_output}${server_help_error}")
  endif()
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}"
          -S "${Q3X_SOURCE_DIR}/tests/package_consumer"
          -B "${consumer_binary}"
          -G "${Q3X_GENERATOR}"
          "-DCMAKE_BUILD_TYPE=${Q3X_CONFIG}"
          "-DCMAKE_PREFIX_PATH=${prefix}"
  RESULT_VARIABLE configure_result
  OUTPUT_VARIABLE configure_output
  ERROR_VARIABLE configure_error
)
if(NOT configure_result EQUAL 0)
  message(FATAL_ERROR
    "consumer configure failed (${configure_result})\n${configure_output}${configure_error}")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" --build "${consumer_binary}"
          --config "${Q3X_CONFIG}"
  RESULT_VARIABLE build_result
  OUTPUT_VARIABLE build_output
  ERROR_VARIABLE build_error
)
if(NOT build_result EQUAL 0)
  message(FATAL_ERROR
    "consumer build failed (${build_result})\n${build_output}${build_error}")
endif()

execute_process(
  COMMAND "${CMAKE_CTEST_COMMAND}" --test-dir "${consumer_binary}"
          --build-config "${Q3X_CONFIG}" --output-on-failure
  RESULT_VARIABLE test_result
  OUTPUT_VARIABLE test_output
  ERROR_VARIABLE test_error
)
if(NOT test_result EQUAL 0)
  message(FATAL_ERROR
    "consumer execution failed (${test_result})\n${test_output}${test_error}")
endif()
