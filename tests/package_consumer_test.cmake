if(NOT DEFINED Q3X_SOURCE_DIR OR NOT DEFINED Q3X_BINARY_DIR OR
   NOT DEFINED Q3X_GENERATOR)
  message(FATAL_ERROR "package consumer test is missing required paths")
endif()

set(prefix "${Q3X_BINARY_DIR}/package-consumer-prefix")
set(consumer_binary "${Q3X_BINARY_DIR}/package-consumer-build")

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
