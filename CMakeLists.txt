cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(Qwen3xOrin VERSION 0.4.0 LANGUAGES CXX CUDA)

include(GNUInstallDirs)
include(CTest)
include(CMakePackageConfigHelpers)
include(cmake/CompilerWarnings.cmake)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(Q3X_CUDA_ARCHITECTURES "87" CACHE STRING
    "CUDA architectures for Qwen3x-Orin (87 is Jetson AGX Orin)")
set(CMAKE_CUDA_ARCHITECTURES "${Q3X_CUDA_ARCHITECTURES}")
option(Q3X_BUILD_GDN_B8_ADMISSION
  "Build the rejected/test-only Prefill GDN B8 real-checkpoint route"
  OFF
)
if(Q3X_BUILD_GDN_B8_ADMISSION AND NOT BUILD_TESTING)
  message(FATAL_ERROR
    "Q3X_BUILD_GDN_B8_ADMISSION is test-only and requires BUILD_TESTING=ON"
  )
endif()

find_package(CUDAToolkit 12.0 REQUIRED)
find_package(ICU 74 REQUIRED COMPONENTS uc i18n)
find_package(Threads REQUIRED)
if(NOT ICU_VERSION MATCHES "^74\\.")
  message(FATAL_ERROR
    "q3x_text is validated against ICU 74.x; found ICU ${ICU_VERSION}")
endif()

configure_file(
  include/q3x/version.h.in
  generated/q3x/version.h
  @ONLY
)
configure_package_config_file(
  cmake/Qwen3xOrinConfig.cmake.in
  generated/Qwen3xOrinConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Qwen3xOrin
)
write_basic_package_version_file(
  generated/Qwen3xOrinConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  # The public ABI is not stable during the 0.x bootstrap series. Consumers
  # must request the exact package version until an explicit compatibility
  # policy is published.
  COMPATIBILITY ExactVersion
)

add_library(q3x_core STATIC
  src/core/device_info.cpp
  src/core/sha256.cpp
  src/io/json.cpp
  src/io/safetensors.cpp
  src/model/checkpoint_metadata.cpp
  src/model/model_config.cpp
  src/model/weight_manifest.cpp
  src/runtime/model_weights.cpp
  src/runtime/reference_oracle.cpp
  src/runtime/resident_weights.cpp
  src/runtime/request_state.cpp
  src/kernels/reference/gemv_reference.cpp
  src/runtime/reference/decode_ops.cpp
  src/runtime/reference/gdn_decode.cpp
  src/runtime/reference/layout_ops.cpp
  src/quantization/fp8.cpp
  src/quantization/nvfp4.cpp
)
add_library(q3x::core ALIAS q3x_core)
set_target_properties(q3x_core PROPERTIES
  EXPORT_NAME core
  POSITION_INDEPENDENT_CODE ON
)
target_compile_features(q3x_core PUBLIC cxx_std_17 cuda_std_17)
target_include_directories(q3x_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(q3x_core PUBLIC CUDA::cudart Threads::Threads)
q3x_enable_warnings(q3x_core)

add_library(q3x_text STATIC
  src/text/tokenizer.cpp
)
add_library(q3x::text ALIAS q3x_text)
set_target_properties(q3x_text PROPERTIES
  EXPORT_NAME text
  POSITION_INDEPENDENT_CODE ON
)
target_compile_features(q3x_text PUBLIC cxx_std_17)
target_include_directories(q3x_text
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
# q3x_text reuses the bounded JSON/SHA implementation from q3x_core. ICU stays
# isolated on this target and is not part of q3x_core's public link interface.
target_link_libraries(q3x_text
  PRIVATE q3x::core
  PUBLIC ICU::uc ICU::i18n
)
q3x_enable_warnings(q3x_text)

add_library(q3x_kernels STATIC
  src/runtime/reference_runner.cpp
  src/kernels/reference/decode_ops.cu
  src/kernels/reference/gdn_decode.cu
  src/kernels/reference/gdn_m16_register_state.cu
  src/kernels/reference/gdn_prefill_b8_sequential_sm87.cu
  src/kernels/reference/layout_ops.cu
  src/kernels/reference/fp8_reference.cu
  src/kernels/reference/gemv_reference.cu
  src/kernels/reference/nvfp4_reference.cu
  src/kernels/reference/projection_dispatch.cpp
  src/kernels/sm87/device_probe.cu
  src/kernels/sm87/weight_only_gemv.cu
)
add_library(q3x::kernels ALIAS q3x_kernels)
set_target_properties(q3x_kernels PROPERTIES
  EXPORT_NAME kernels
  POSITION_INDEPENDENT_CODE ON
)
target_compile_features(q3x_kernels PUBLIC cxx_std_17 cuda_std_17)
target_include_directories(q3x_kernels
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(q3x_kernels PUBLIC q3x::core CUDA::cudart)
target_compile_options(q3x_kernels PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
if(Q3X_BUILD_GDN_B8_ADMISSION)
  target_compile_definitions(q3x_kernels PRIVATE
    Q3X_ENABLE_GDN_B8_ADMISSION=1
  )
endif()
q3x_enable_warnings(q3x_kernels)

add_library(q3x_engine STATIC
  src/runtime/reference_benchmark.cpp
  src/runtime/reference_engine.cpp
  src/runtime/reference_engine_control.cpp
)
add_library(q3x::engine ALIAS q3x_engine)
set_target_properties(q3x_engine PROPERTIES
  EXPORT_NAME engine
  POSITION_INDEPENDENT_CODE ON
)
target_compile_features(q3x_engine PUBLIC cxx_std_17)
target_include_directories(q3x_engine
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
# The public engine API exposes runner/core result types. The tokenizer is an
# implementation dependency; CMake still propagates it as LINK_ONLY when this
# static library is consumed or exported.
target_link_libraries(q3x_engine
  PUBLIC q3x::core q3x::kernels
  PRIVATE q3x::text
)
q3x_enable_warnings(q3x_engine)

add_executable(qwen3x-orin src/main.cpp)
target_link_libraries(qwen3x-orin
  PRIVATE q3x::engine q3x::text q3x::core q3x::kernels
)
q3x_enable_warnings(qwen3x-orin)

add_executable(qwen3x-inspect tools/q3x-inspect/main.cpp)
target_link_libraries(qwen3x-inspect PRIVATE q3x::core)
q3x_enable_warnings(qwen3x-inspect)

if(BUILD_TESTING)
  add_executable(q3x_device_info_test tests/device_info_test.cpp)
  target_link_libraries(q3x_device_info_test PRIVATE q3x::core q3x::kernels)
  q3x_enable_warnings(q3x_device_info_test)
  add_test(NAME device_info COMMAND q3x_device_info_test)
  set_tests_properties(device_info PROPERTIES SKIP_RETURN_CODE 77)

  add_executable(q3x_nvfp4_test tests/nvfp4_test.cu)
  target_link_libraries(q3x_nvfp4_test PRIVATE q3x::core q3x::kernels)
  target_compile_options(q3x_nvfp4_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  )
  q3x_enable_warnings(q3x_nvfp4_test)
  add_test(NAME nvfp4_reference COMMAND q3x_nvfp4_test)

  add_executable(q3x_sm87_weight_only_gemv_test
    tests/sm87_weight_only_gemv_test.cu
  )
  target_link_libraries(q3x_sm87_weight_only_gemv_test
    PRIVATE q3x::kernels
  )
  q3x_enable_warnings(q3x_sm87_weight_only_gemv_test)
  add_test(NAME sm87_weight_only_gemv
    COMMAND q3x_sm87_weight_only_gemv_test
  )
  set_tests_properties(sm87_weight_only_gemv PROPERTIES
    LABELS "cuda;sm87"
    SKIP_RETURN_CODE 77
  )

  # Isolated Gate/Up activation-persistence screen. This target reuses the
  # public production M128 launcher while keeping all L2 access-policy-window
  # experiments outside the production runner, dispatch, and kernel library.
  add_executable(q3x_nvfp4_prefill_gate_up_l2_persist_screen
    tests/sm87_nvfp4_prefill_gate_up_l2_persist_screen.cu
  )
  target_link_libraries(q3x_nvfp4_prefill_gate_up_l2_persist_screen
    PRIVATE q3x::kernels
  )
  q3x_enable_warnings(q3x_nvfp4_prefill_gate_up_l2_persist_screen)
  add_test(NAME nvfp4_prefill_gate_up_l2_persist_screen
    COMMAND q3x_nvfp4_prefill_gate_up_l2_persist_screen
  )
  add_test(NAME nvfp4_prefill_gate_up_l2_persist_screen_c256
    COMMAND q3x_nvfp4_prefill_gate_up_l2_persist_screen --tokens=256
  )
  set_tests_properties(
    nvfp4_prefill_gate_up_l2_persist_screen
    nvfp4_prefill_gate_up_l2_persist_screen_c256
    PROPERTIES
    LABELS "cuda;sm87;prefill"
    SKIP_RETURN_CODE 77
  )

  # Isolated canonical-layout Gate/Up pair candidate. One 512-thread CTA
  # shares the M128 activation tile between two independent eight-warp branch
  # groups; production kernels, dispatch, and runner scheduling stay frozen.
  add_executable(q3x_nvfp4_prefill_gate_up_fused_m128_shared_a_screen
    tests/sm87_nvfp4_prefill_gate_up_fused_m128_shared_a_screen.cu
  )
  target_link_libraries(
    q3x_nvfp4_prefill_gate_up_fused_m128_shared_a_screen
    PRIVATE q3x::kernels
  )
  q3x_enable_warnings(
    q3x_nvfp4_prefill_gate_up_fused_m128_shared_a_screen
  )
  add_test(NAME nvfp4_prefill_gate_up_fused_m128_shared_a_screen
    COMMAND q3x_nvfp4_prefill_gate_up_fused_m128_shared_a_screen
  )
  add_test(NAME nvfp4_prefill_gate_up_fused_m128_shared_a_screen_c256
    COMMAND q3x_nvfp4_prefill_gate_up_fused_m128_shared_a_screen --tokens=256
  )
  set_tests_properties(
    nvfp4_prefill_gate_up_fused_m128_shared_a_screen
    nvfp4_prefill_gate_up_fused_m128_shared_a_screen_c256
    PROPERTIES
    LABELS "cuda;sm87;prefill"
    SKIP_RETURN_CODE 77
  )

  # Isolated FP8 QKV latency-hiding candidate. The production M128xN128
  # route remains frozen; this target only screens a 512-thread split-M64
  # warp-group implementation for exact C512 [10240,5120].
  add_executable(q3x_fp8_prefill_qkv_m128_split_m64_screen
    tests/sm87_fp8_prefill_qkv_m128_split_m64_screen.cu
  )
  target_link_libraries(q3x_fp8_prefill_qkv_m128_split_m64_screen
    PRIVATE q3x::kernels
  )
  q3x_enable_warnings(q3x_fp8_prefill_qkv_m128_split_m64_screen)
  add_test(NAME fp8_prefill_qkv_m128_split_m64_screen
    COMMAND q3x_fp8_prefill_qkv_m128_split_m64_screen
  )
  set_tests_properties(fp8_prefill_qkv_m128_split_m64_screen PROPERTIES
    LABELS "cuda;sm87;prefill"
    SKIP_RETURN_CODE 77
  )

  # Production FP8 QKV C512 dataflow sentinel. This target validates the GPU
  # fragment-native sidecar pack, three-stage raw A/B cp.async pipeline, and
  # direct matrix_b register feed independently of the full model runner.
  add_executable(q3x_fp8_prefill_qkv_m128_cp_async_register_feed_screen
    tests/sm87_fp8_prefill_qkv_m128_cp_async_register_feed_screen.cu
  )
  target_link_libraries(
    q3x_fp8_prefill_qkv_m128_cp_async_register_feed_screen
    PRIVATE q3x::kernels
  )
  q3x_enable_warnings(
    q3x_fp8_prefill_qkv_m128_cp_async_register_feed_screen
  )
  add_test(NAME fp8_prefill_qkv_m128_cp_async_register_feed_screen
    COMMAND q3x_fp8_prefill_qkv_m128_cp_async_register_feed_screen
  )
  add_test(NAME fp8_prefill_qkv_m128_cp_async_register_feed_canonical_validate
    COMMAND q3x_fp8_prefill_qkv_m128_cp_async_register_feed_screen
            --mode=validate-canonical
  )
  set_tests_properties(
    fp8_prefill_qkv_m128_cp_async_register_feed_screen
    fp8_prefill_qkv_m128_cp_async_register_feed_canonical_validate
    PROPERTIES
    LABELS "cuda;sm87;prefill"
    SKIP_RETURN_CODE 77
  )

  # Test-only SM87 sentinel for the lane/slot layout used when a BF16 WMMA
  # matrix_b fragment is fed directly from registers instead of shared memory.
  add_executable(q3x_sm87_bf16_matrix_b_register_feed_sentinel
    tests/sm87_bf16_matrix_b_register_feed_sentinel.cu
  )
  target_link_libraries(q3x_sm87_bf16_matrix_b_register_feed_sentinel
    PRIVATE CUDA::cudart
  )
  target_compile_options(q3x_sm87_bf16_matrix_b_register_feed_sentinel PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
  )
  q3x_enable_warnings(q3x_sm87_bf16_matrix_b_register_feed_sentinel)
  add_test(NAME sm87_bf16_matrix_b_register_feed_sentinel
    COMMAND q3x_sm87_bf16_matrix_b_register_feed_sentinel
  )
  set_tests_properties(sm87_bf16_matrix_b_register_feed_sentinel PROPERTIES
    LABELS "cuda;sm87;prefill"
    SKIP_RETURN_CODE 77
  )

  # Isolated first-cell screen for the fragment-native, register-fed NVFP4
  # Gate candidate. Sidecar construction and layout validation remain test
  # setup; neither the production dispatch nor the timed launch includes it.
  add_executable(q3x_nvfp4_prefill_gate_register_fed_screen
    tests/sm87_nvfp4_prefill_gate_register_fed_screen.cu
  )
  target_link_libraries(q3x_nvfp4_prefill_gate_register_fed_screen
    PRIVATE q3x::kernels
  )
  q3x_enable_warnings(q3x_nvfp4_prefill_gate_register_fed_screen)
  add_test(NAME nvfp4_prefill_gate_register_fed_screen
    COMMAND q3x_nvfp4_prefill_gate_register_fed_screen
  )
  add_test(NAME nvfp4_prefill_gate_register_fed_screen_c256
    COMMAND q3x_nvfp4_prefill_gate_register_fed_screen --tokens=256
  )
  set_tests_properties(
    nvfp4_prefill_gate_register_fed_screen
    nvfp4_prefill_gate_register_fed_screen_c256
    PROPERTIES
    LABELS "cuda;sm87;prefill"
    SKIP_RETURN_CODE 77
  )

  add_executable(q3x_sm87_projection_route_registry_test
    tests/sm87_projection_route_registry_test.cpp
  )
  target_include_directories(q3x_sm87_projection_route_registry_test
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
  )
  q3x_enable_warnings(q3x_sm87_projection_route_registry_test)
  add_test(NAME sm87_projection_route_registry
    COMMAND q3x_sm87_projection_route_registry_test
  )

  add_executable(q3x_projection_backend_dispatch_test
    tests/projection_backend_dispatch_test.cu
  )
  target_link_libraries(q3x_projection_backend_dispatch_test
    PRIVATE q3x::kernels
  )
  q3x_enable_warnings(q3x_projection_backend_dispatch_test)
  add_test(NAME projection_backend_dispatch
    COMMAND q3x_projection_backend_dispatch_test
  )
  set_tests_properties(projection_backend_dispatch PROPERTIES
    LABELS "cuda;sm87"
    SKIP_RETURN_CODE 77
  )

  add_executable(q3x_fp8_test tests/fp8_test.cu)
  target_link_libraries(q3x_fp8_test PRIVATE q3x::core q3x::kernels)
  target_compile_options(q3x_fp8_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  )
  q3x_enable_warnings(q3x_fp8_test)
  add_test(NAME fp8_reference COMMAND q3x_fp8_test)

  add_executable(q3x_reference_gemv_host_test
    tests/reference_gemv_host_test.cpp
  )
  target_link_libraries(q3x_reference_gemv_host_test PRIVATE q3x::core)
  q3x_enable_warnings(q3x_reference_gemv_host_test)
  add_test(NAME reference_gemv_host COMMAND q3x_reference_gemv_host_test)

  add_executable(q3x_reference_gemv_cuda_test
    tests/reference_gemv_cuda_test.cu
  )
  target_link_libraries(q3x_reference_gemv_cuda_test
    PRIVATE q3x::core q3x::kernels
  )
  target_compile_options(q3x_reference_gemv_cuda_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  )
  q3x_enable_warnings(q3x_reference_gemv_cuda_test)
  add_test(NAME reference_gemv_cuda COMMAND q3x_reference_gemv_cuda_test)

  add_executable(q3x_decode_ops_host_test
    tests/decode_ops_host_test.cpp
  )
  target_link_libraries(q3x_decode_ops_host_test PRIVATE q3x::core)
  q3x_enable_warnings(q3x_decode_ops_host_test)
  add_test(NAME decode_ops_host COMMAND q3x_decode_ops_host_test)

  add_executable(q3x_decode_ops_cuda_test
    tests/decode_ops_cuda_test.cu
  )
  target_link_libraries(q3x_decode_ops_cuda_test
    PRIVATE q3x::core q3x::kernels
  )
  target_compile_options(q3x_decode_ops_cuda_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  )
  q3x_enable_warnings(q3x_decode_ops_cuda_test)
  add_test(NAME decode_ops_cuda COMMAND q3x_decode_ops_cuda_test)

  add_executable(q3x_gdn_decode_host_test
    tests/gdn_decode_host_test.cpp
  )
  target_link_libraries(q3x_gdn_decode_host_test PRIVATE q3x::core)
  q3x_enable_warnings(q3x_gdn_decode_host_test)
  add_test(NAME gdn_decode_host COMMAND q3x_gdn_decode_host_test)

  add_executable(q3x_gdn_decode_cuda_test
    tests/gdn_decode_cuda_test.cu
    tests/gdn_prefill_whole_span_sm87_candidate.cu
  )
  target_link_libraries(q3x_gdn_decode_cuda_test
    PRIVATE q3x::core q3x::kernels
  )
  target_compile_options(q3x_gdn_decode_cuda_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  )
  q3x_enable_warnings(q3x_gdn_decode_cuda_test)
  add_test(NAME gdn_decode_cuda COMMAND q3x_gdn_decode_cuda_test)

  # Isolated Prefill block-transition selection screen. Its local oracle/WY
  # CUDA templates remain test-only; the separately admitted sequential
  # private launcher is linked from q3x_kernels for equivalence checks.
  add_executable(q3x_gdn_prefill_b8_block_sm87_screen
    tests/gdn_prefill_b8_block_sm87_screen.cu
  )
  target_link_libraries(q3x_gdn_prefill_b8_block_sm87_screen
    PRIVATE q3x::core q3x::kernels
  )
  target_compile_options(q3x_gdn_prefill_b8_block_sm87_screen PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-Xptxas=-v>
  )
  q3x_enable_warnings(q3x_gdn_prefill_b8_block_sm87_screen)
  add_test(NAME gdn_prefill_b8_block_sm87_screen
    COMMAND q3x_gdn_prefill_b8_block_sm87_screen
  )
  set_tests_properties(gdn_prefill_b8_block_sm87_screen PROPERTIES
    LABELS "cuda;sm87;prefill"
    SKIP_RETURN_CODE 77
  )

  # Isolated exact-C16 scalar-vector/QK ping-pong screen. The candidate kernel
  # is compiled into this test executable only and cannot affect production
  # selectors, public ABI, or q3x_kernels SASS.
  add_executable(q3x_gdn_prefill_c16_pingpong_sm87_candidate
    tests/gdn_prefill_c16_pingpong_sm87_candidate.cu
  )
  target_link_libraries(q3x_gdn_prefill_c16_pingpong_sm87_candidate
    PRIVATE q3x::core q3x::kernels
  )
  target_compile_options(q3x_gdn_prefill_c16_pingpong_sm87_candidate PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-Xptxas=-v>
  )
  q3x_enable_warnings(q3x_gdn_prefill_c16_pingpong_sm87_candidate)
  add_test(NAME gdn_prefill_c16_pingpong_sm87_candidate
    COMMAND q3x_gdn_prefill_c16_pingpong_sm87_candidate
  )
  set_tests_properties(gdn_prefill_c16_pingpong_sm87_candidate PROPERTIES
    LABELS "cuda;sm87;prefill"
    SKIP_RETURN_CODE 77
  )

  add_executable(q3x_layout_ops_host_test
    tests/layout_ops_host_test.cpp
  )
  target_link_libraries(q3x_layout_ops_host_test PRIVATE q3x::core)
  q3x_enable_warnings(q3x_layout_ops_host_test)
  add_test(NAME layout_ops_host COMMAND q3x_layout_ops_host_test)

  add_executable(q3x_layout_ops_cuda_test
    tests/layout_ops_cuda_test.cu
  )
  target_link_libraries(q3x_layout_ops_cuda_test
    PRIVATE q3x::core q3x::kernels
  )
  target_compile_options(q3x_layout_ops_cuda_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  )
  q3x_enable_warnings(q3x_layout_ops_cuda_test)
  add_test(NAME layout_ops_cuda COMMAND q3x_layout_ops_cuda_test)

  add_executable(q3x_reference_runner_host_test
    tests/reference_runner_host_test.cpp
  )
  target_link_libraries(q3x_reference_runner_host_test PRIVATE q3x::kernels)
  q3x_enable_warnings(q3x_reference_runner_host_test)
  add_test(NAME reference_runner_host COMMAND q3x_reference_runner_host_test)

  add_executable(q3x_reference_engine_control_test
    tests/reference_engine_control_test.cpp
  )
  target_link_libraries(q3x_reference_engine_control_test
    PRIVATE q3x::engine
  )
  q3x_enable_warnings(q3x_reference_engine_control_test)
  add_test(NAME reference_engine_control
    COMMAND q3x_reference_engine_control_test
  )

  add_executable(q3x_reference_benchmark_control_test
    tests/reference_benchmark_control_test.cpp
  )
  target_link_libraries(q3x_reference_benchmark_control_test
    PRIVATE q3x::engine
  )
  q3x_enable_warnings(q3x_reference_benchmark_control_test)
  add_test(NAME reference_benchmark_control
    COMMAND q3x_reference_benchmark_control_test
  )

  set(Q3X_E2E_MODEL_DIR "" CACHE PATH
      "Pinned Qwen3.6-27B-NVFP4 model directory for the optional E2E test")
  set(Q3X_E2E_PROJECTION_BACKEND "reference" CACHE STRING
      "Projection backend for the optional E2E test (reference or sm87)")
  set_property(CACHE Q3X_E2E_PROJECTION_BACKEND PROPERTY STRINGS
      reference sm87)
  if(NOT Q3X_E2E_PROJECTION_BACKEND STREQUAL "reference" AND
     NOT Q3X_E2E_PROJECTION_BACKEND STREQUAL "sm87")
    message(FATAL_ERROR
      "Q3X_E2E_PROJECTION_BACKEND must be 'reference' or 'sm87'")
  endif()
  set(_q3x_e2e_model_dir "${Q3X_E2E_MODEL_DIR}")
  if(NOT _q3x_e2e_model_dir AND DEFINED ENV{Q3X_E2E_MODEL_DIR})
    set(_q3x_e2e_model_dir "$ENV{Q3X_E2E_MODEL_DIR}")
  endif()
  if(NOT _q3x_e2e_model_dir)
    set(_q3x_e2e_model_dir "-")
  endif()

  add_executable(q3x_reference_engine_e2e_test
    tests/reference_engine_e2e_test.cpp
  )
  target_link_libraries(q3x_reference_engine_e2e_test
    PRIVATE q3x::engine
  )
  q3x_enable_warnings(q3x_reference_engine_e2e_test)
  add_test(NAME reference_engine_e2e
    COMMAND q3x_reference_engine_e2e_test "${_q3x_e2e_model_dir}"
            "${Q3X_E2E_PROJECTION_BACKEND}"
  )
  set_tests_properties(reference_engine_e2e PROPERTIES
    LABELS "e2e;model"
    RUN_SERIAL TRUE
    SKIP_RETURN_CODE 77
    TIMEOUT 900
  )
  add_test(NAME reference_engine_e2e_chunk8
    COMMAND q3x_reference_engine_e2e_test "${_q3x_e2e_model_dir}"
            "${Q3X_E2E_PROJECTION_BACKEND}" "8"
  )
  set_tests_properties(reference_engine_e2e_chunk8 PROPERTIES
    LABELS "e2e;model"
    RUN_SERIAL TRUE
    SKIP_RETURN_CODE 77
    TIMEOUT 900
  )
  add_test(NAME reference_engine_e2e_chunk16
    COMMAND q3x_reference_engine_e2e_test "${_q3x_e2e_model_dir}"
            "${Q3X_E2E_PROJECTION_BACKEND}" "16"
  )
  set_tests_properties(reference_engine_e2e_chunk16 PROPERTIES
    LABELS "e2e;model"
    RUN_SERIAL TRUE
    SKIP_RETURN_CODE 77
    TIMEOUT 900
  )
  add_test(NAME reference_engine_e2e_chunk32
    COMMAND q3x_reference_engine_e2e_test "${_q3x_e2e_model_dir}"
            "${Q3X_E2E_PROJECTION_BACKEND}" "32"
  )
  set_tests_properties(reference_engine_e2e_chunk32 PROPERTIES
    LABELS "e2e;model"
    RUN_SERIAL TRUE
    SKIP_RETURN_CODE 77
    TIMEOUT 900
  )
  add_test(NAME reference_engine_e2e_chunk64
    COMMAND q3x_reference_engine_e2e_test "${_q3x_e2e_model_dir}"
            "${Q3X_E2E_PROJECTION_BACKEND}" "64"
  )
  set_tests_properties(reference_engine_e2e_chunk64 PROPERTIES
    LABELS "e2e;model"
    RUN_SERIAL TRUE
    SKIP_RETURN_CODE 77
    TIMEOUT 900
  )
  add_test(NAME reference_engine_e2e_chunk256
    COMMAND q3x_reference_engine_e2e_test "${_q3x_e2e_model_dir}"
            "${Q3X_E2E_PROJECTION_BACKEND}" "256"
  )
  set_tests_properties(reference_engine_e2e_chunk256 PROPERTIES
    LABELS "e2e;model"
    RUN_SERIAL TRUE
    SKIP_RETURN_CODE 77
    TIMEOUT 900
  )
  add_test(NAME reference_engine_e2e_chunk512
    COMMAND q3x_reference_engine_e2e_test "${_q3x_e2e_model_dir}"
            "${Q3X_E2E_PROJECTION_BACKEND}" "512"
  )
  set_tests_properties(reference_engine_e2e_chunk512 PROPERTIES
    LABELS "e2e;model"
    RUN_SERIAL TRUE
    SKIP_RETURN_CODE 77
    TIMEOUT 900
  )
  add_executable(q3x_reference_prefill_bulk_attention_e2e_test
    tests/reference_prefill_bulk_attention_e2e_test.cpp
  )
  target_link_libraries(q3x_reference_prefill_bulk_attention_e2e_test
    PRIVATE q3x::engine
  )
  q3x_enable_warnings(q3x_reference_prefill_bulk_attention_e2e_test)
  add_test(NAME reference_prefill_bulk_attention_e2e
    COMMAND q3x_reference_prefill_bulk_attention_e2e_test
            "${_q3x_e2e_model_dir}"
  )
  set_tests_properties(reference_prefill_bulk_attention_e2e PROPERTIES
    LABELS "e2e;model;sm87;prefill;bulk_attention"
    RUN_SERIAL TRUE
    SKIP_RETURN_CODE 77
    TIMEOUT 1800
  )

  if(Q3X_BUILD_GDN_B8_ADMISSION)
    add_executable(q3x_reference_gdn_prefill_b8_admission_e2e_test
      tests/reference_gdn_prefill_b8_admission_e2e_test.cpp
    )
    target_link_libraries(q3x_reference_gdn_prefill_b8_admission_e2e_test
      PRIVATE q3x::kernels
    )
    q3x_enable_warnings(q3x_reference_gdn_prefill_b8_admission_e2e_test)
    add_test(NAME reference_gdn_prefill_b8_admission_e2e
      COMMAND q3x_reference_gdn_prefill_b8_admission_e2e_test
              "${_q3x_e2e_model_dir}"
    )
    set_tests_properties(reference_gdn_prefill_b8_admission_e2e PROPERTIES
      LABELS "e2e;model;sm87;prefill;gdn;admission"
      RUN_SERIAL TRUE
      SKIP_RETURN_CODE 77
      TIMEOUT 1800
    )

    add_executable(q3x_reference_gdn_prefill_b8_engine_e2e_test
      tests/reference_gdn_prefill_b8_engine_e2e_test.cpp
    )
    target_link_libraries(q3x_reference_gdn_prefill_b8_engine_e2e_test
      PRIVATE q3x::engine
    )
    q3x_enable_warnings(q3x_reference_gdn_prefill_b8_engine_e2e_test)
    add_test(NAME reference_gdn_prefill_b8_engine_e2e
      COMMAND q3x_reference_gdn_prefill_b8_engine_e2e_test
              "${_q3x_e2e_model_dir}"
    )
    set_tests_properties(reference_gdn_prefill_b8_engine_e2e PROPERTIES
      LABELS "e2e;model;sm87;prefill;gdn;admission;ability"
      RUN_SERIAL TRUE
      SKIP_RETURN_CODE 77
      TIMEOUT 1800
    )
  endif()

  add_executable(q3x_reference_decode_graph_p1_perf_test
    tests/reference_decode_graph_p1_perf_test.cpp
  )
  target_link_libraries(q3x_reference_decode_graph_p1_perf_test
    PRIVATE q3x::engine
  )
  q3x_enable_warnings(q3x_reference_decode_graph_p1_perf_test)
  add_test(NAME reference_decode_graph_p1_perf
    COMMAND q3x_reference_decode_graph_p1_perf_test
            "${_q3x_e2e_model_dir}"
  )
  set_tests_properties(reference_decode_graph_p1_perf PROPERTIES
    LABELS "e2e;model;performance;decode;cuda_graph"
    RUN_SERIAL TRUE
    SKIP_RETURN_CODE 77
    TIMEOUT 1800
  )

  add_executable(q3x_reference_decode_graph_p2_perf_test
    tests/reference_decode_graph_p2_perf_test.cpp
  )
  target_link_libraries(q3x_reference_decode_graph_p2_perf_test
    PRIVATE q3x::engine
  )
  q3x_enable_warnings(q3x_reference_decode_graph_p2_perf_test)
  add_test(NAME reference_decode_graph_p2_perf
    COMMAND q3x_reference_decode_graph_p2_perf_test
            "${_q3x_e2e_model_dir}"
  )
  set_tests_properties(reference_decode_graph_p2_perf PROPERTIES
    LABELS "e2e;model;performance;decode;cuda_graph"
    RUN_SERIAL TRUE
    SKIP_RETURN_CODE 77
    TIMEOUT 3600
  )

  add_executable(q3x_reference_decode_graph_production_test
    tests/reference_decode_graph_production_test.cpp
  )
  target_link_libraries(q3x_reference_decode_graph_production_test
    PRIVATE q3x::engine
  )
  q3x_enable_warnings(q3x_reference_decode_graph_production_test)
  add_test(NAME reference_decode_graph_production
    COMMAND q3x_reference_decode_graph_production_test
            "${_q3x_e2e_model_dir}"
  )
  set_tests_properties(reference_decode_graph_production PROPERTIES
    LABELS "e2e;model;decode;cuda_graph;production"
    RUN_SERIAL TRUE
    SKIP_RETURN_CODE 77
    TIMEOUT 3600
  )

  add_executable(q3x_model_config_test tests/model_config_test.cpp)
  target_link_libraries(q3x_model_config_test PRIVATE q3x::core)
  q3x_enable_warnings(q3x_model_config_test)
  add_test(NAME model_config COMMAND q3x_model_config_test)

  add_executable(q3x_checkpoint_metadata_test
    tests/checkpoint_metadata_test.cpp
  )
  target_link_libraries(q3x_checkpoint_metadata_test PRIVATE q3x::core)
  q3x_enable_warnings(q3x_checkpoint_metadata_test)
  add_test(NAME checkpoint_metadata COMMAND q3x_checkpoint_metadata_test)

  add_executable(q3x_weight_manifest_test
    tests/weight_manifest_test.cpp
  )
  target_link_libraries(q3x_weight_manifest_test PRIVATE q3x::core)
  q3x_enable_warnings(q3x_weight_manifest_test)
  add_test(NAME weight_manifest COMMAND q3x_weight_manifest_test)

  add_executable(q3x_resident_load_plan_test
    tests/resident_load_plan_test.cpp
  )
  target_link_libraries(q3x_resident_load_plan_test PRIVATE q3x::core)
  q3x_enable_warnings(q3x_resident_load_plan_test)
  add_test(NAME resident_load_plan COMMAND q3x_resident_load_plan_test)

  add_executable(q3x_request_state_plan_test
    tests/request_state_plan_test.cpp
  )
  target_link_libraries(q3x_request_state_plan_test PRIVATE q3x::core)
  q3x_enable_warnings(q3x_request_state_plan_test)
  add_test(NAME request_state_plan COMMAND q3x_request_state_plan_test)

  add_executable(q3x_request_state_cuda_test
    tests/request_state_cuda_test.cpp
  )
  target_link_libraries(q3x_request_state_cuda_test PRIVATE q3x::core)
  q3x_enable_warnings(q3x_request_state_cuda_test)
  add_test(NAME request_state_cuda COMMAND q3x_request_state_cuda_test)
  set_tests_properties(request_state_cuda PROPERTIES SKIP_RETURN_CODE 77)

  add_executable(q3x_resident_weights_cuda_test
    tests/resident_weights_cuda_test.cpp
  )
  target_link_libraries(q3x_resident_weights_cuda_test PRIVATE q3x::core)
  q3x_enable_warnings(q3x_resident_weights_cuda_test)
  add_test(NAME resident_weights_cuda COMMAND q3x_resident_weights_cuda_test)
  set_tests_properties(resident_weights_cuda PROPERTIES SKIP_RETURN_CODE 77)

  add_executable(q3x_model_weights_host_test
    tests/model_weights_host_test.cpp
  )
  target_link_libraries(q3x_model_weights_host_test PRIVATE q3x::core)
  q3x_enable_warnings(q3x_model_weights_host_test)
  add_test(NAME model_weights_host COMMAND q3x_model_weights_host_test)

  add_executable(q3x_reference_oracle_test
    tests/reference_oracle_test.cpp
  )
  target_link_libraries(q3x_reference_oracle_test PRIVATE q3x::core)
  q3x_enable_warnings(q3x_reference_oracle_test)
  add_test(NAME reference_oracle COMMAND q3x_reference_oracle_test
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/qwen36-27b-nvfp4-greedy-bf16.json"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/qwen36-27b-nvfp4-layers-bf16.json"
  )

  add_executable(q3x_model_weights_cuda_test
    tests/model_weights_cuda_test.cu
  )
  target_link_libraries(q3x_model_weights_cuda_test
    PRIVATE q3x::core q3x::kernels
  )
  target_compile_options(q3x_model_weights_cuda_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  )
  q3x_enable_warnings(q3x_model_weights_cuda_test)
  add_test(NAME model_weights_cuda COMMAND q3x_model_weights_cuda_test)
  set_tests_properties(model_weights_cuda PROPERTIES SKIP_RETURN_CODE 77)

  add_executable(q3x_sha256_test tests/sha256_test.cpp)
  target_link_libraries(q3x_sha256_test PRIVATE q3x::core)
  q3x_enable_warnings(q3x_sha256_test)
  add_test(NAME sha256 COMMAND q3x_sha256_test)

  add_executable(q3x_json_test tests/json_test.cpp)
  target_link_libraries(q3x_json_test PRIVATE q3x::core)
  q3x_enable_warnings(q3x_json_test)
  add_test(NAME json COMMAND q3x_json_test)

  add_executable(q3x_safetensors_test tests/safetensors_test.cpp)
  target_link_libraries(q3x_safetensors_test PRIVATE q3x::core)
  q3x_enable_warnings(q3x_safetensors_test)
  add_test(NAME safetensors COMMAND q3x_safetensors_test)

  set(Q3X_TOKENIZER_TEST_JSON "" CACHE FILEPATH
      "Pinned Qwen 3.6 tokenizer.json used by tokenizer differential tests")
  set(_q3x_tokenizer_test_json "${Q3X_TOKENIZER_TEST_JSON}")
  if(NOT _q3x_tokenizer_test_json AND
     DEFINED ENV{Q3X_TOKENIZER_TEST_JSON})
    set(_q3x_tokenizer_test_json "$ENV{Q3X_TOKENIZER_TEST_JSON}")
  endif()
  if(NOT _q3x_tokenizer_test_json)
    set(_q3x_tokenizer_test_json "-")
  endif()

  add_executable(q3x_tokenizer_test tests/tokenizer_test.cpp)
  target_link_libraries(q3x_tokenizer_test PRIVATE q3x::text q3x::core)
  q3x_enable_warnings(q3x_tokenizer_test)
  add_test(NAME tokenizer COMMAND q3x_tokenizer_test
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/qwen36-27b-tokenizer.json"
    "${_q3x_tokenizer_test_json}"
  )
  set_tests_properties(tokenizer PROPERTIES SKIP_RETURN_CODE 77)

  add_test(NAME inspect_help COMMAND qwen3x-inspect help)
  add_test(NAME generate_help COMMAND qwen3x-orin generate --help)
  add_test(NAME generate_invalid_max
    COMMAND qwen3x-orin generate unused-model-directory
            --prompt test --max-tokens 0
  )
  set_tests_properties(generate_invalid_max PROPERTIES WILL_FAIL TRUE)
  add_test(NAME generate_invalid_projection_backend
    COMMAND qwen3x-orin generate unused-model-directory
            --prompt test --projection-backend invalid
  )
  set_tests_properties(generate_invalid_projection_backend
    PROPERTIES WILL_FAIL TRUE
  )
  add_test(NAME generate_duplicate_projection_backend
    COMMAND qwen3x-orin generate unused-model-directory
            --prompt test --projection-backend reference
            --projection-backend sm87
  )
  set_tests_properties(generate_duplicate_projection_backend
    PROPERTIES WILL_FAIL TRUE
  )
  add_test(NAME generate_invalid_prefill_chunk_zero
    COMMAND qwen3x-orin generate unused-model-directory
            --prompt test --prefill-chunk-size 0
  )
  set_tests_properties(generate_invalid_prefill_chunk_zero
    PROPERTIES WILL_FAIL TRUE
  )
  add_test(NAME generate_invalid_prefill_chunk_five_thirteen
    COMMAND qwen3x-orin generate unused-model-directory
            --prompt test --prefill-chunk-size 513
  )
  set_tests_properties(generate_invalid_prefill_chunk_five_thirteen
    PROPERTIES WILL_FAIL TRUE
  )
  add_test(NAME generate_invalid_prefill_chunk_non_numeric
    COMMAND qwen3x-orin generate unused-model-directory
            --prompt test --prefill-chunk-size invalid
  )
  set_tests_properties(generate_invalid_prefill_chunk_non_numeric
    PROPERTIES WILL_FAIL TRUE
  )
  add_test(NAME generate_duplicate_prefill_chunk
    COMMAND qwen3x-orin generate unused-model-directory
            --prompt test --prefill-chunk-size 1
            --prefill-chunk-size 16
  )
  set_tests_properties(generate_duplicate_prefill_chunk
    PROPERTIES WILL_FAIL TRUE
  )
  add_test(NAME generate_duplicate_nvtx_phase_ranges
    COMMAND qwen3x-orin generate unused-model-directory
            --prompt test --nvtx-phase-ranges --nvtx-phase-ranges
  )
  set_tests_properties(generate_duplicate_nvtx_phase_ranges
    PROPERTIES WILL_FAIL TRUE
  )
  add_test(NAME benchmark_help COMMAND qwen3x-orin benchmark --help)
  add_test(NAME benchmark_invalid_iterations
    COMMAND qwen3x-orin benchmark unused-model-directory
            --prompt test --iterations 0
  )
  set_tests_properties(benchmark_invalid_iterations PROPERTIES WILL_FAIL TRUE)
  add_test(NAME benchmark_invalid_projection_backend
    COMMAND qwen3x-orin benchmark unused-model-directory
            --prompt test --projection-backend invalid
  )
  set_tests_properties(benchmark_invalid_projection_backend
    PROPERTIES WILL_FAIL TRUE
  )
  add_test(NAME benchmark_duplicate_projection_backend
    COMMAND qwen3x-orin benchmark unused-model-directory
            --prompt test --projection-backend reference
            --projection-backend sm87
  )
  set_tests_properties(benchmark_duplicate_projection_backend
    PROPERTIES WILL_FAIL TRUE
  )
  add_test(NAME benchmark_invalid_prefill_chunk_zero
    COMMAND qwen3x-orin benchmark unused-model-directory
            --prompt test --prefill-chunk-size 0
  )
  set_tests_properties(benchmark_invalid_prefill_chunk_zero
    PROPERTIES WILL_FAIL TRUE
  )
  add_test(NAME benchmark_invalid_prefill_chunk_five_thirteen
    COMMAND qwen3x-orin benchmark unused-model-directory
            --prompt test --prefill-chunk-size 513
  )
  set_tests_properties(benchmark_invalid_prefill_chunk_five_thirteen
    PROPERTIES WILL_FAIL TRUE
  )
  add_test(NAME benchmark_invalid_prefill_chunk_non_numeric
    COMMAND qwen3x-orin benchmark unused-model-directory
            --prompt test --prefill-chunk-size invalid
  )
  set_tests_properties(benchmark_invalid_prefill_chunk_non_numeric
    PROPERTIES WILL_FAIL TRUE
  )
  add_test(NAME benchmark_duplicate_prefill_chunk
    COMMAND qwen3x-orin benchmark unused-model-directory
            --prompt test --prefill-chunk-size 1
            --prefill-chunk-size 16
  )
  set_tests_properties(benchmark_duplicate_prefill_chunk
    PROPERTIES WILL_FAIL TRUE
  )
  add_test(NAME benchmark_duplicate_nvtx_phase_ranges
    COMMAND qwen3x-orin benchmark unused-model-directory
            --prompt test --nvtx-phase-ranges --nvtx-phase-ranges
  )
  set_tests_properties(benchmark_duplicate_nvtx_phase_ranges
    PROPERTIES WILL_FAIL TRUE
  )
  add_test(NAME benchmark_request_plan_preflight
    COMMAND qwen3x-orin benchmark unused-model-directory
            --prompt test --max-tokens 1
            --max-sequence-length 262144
  )
  set_tests_properties(benchmark_request_plan_preflight
    PROPERTIES WILL_FAIL TRUE
  )

  add_test(NAME package_consumer
    COMMAND ${CMAKE_COMMAND}
      -DQ3X_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
      -DQ3X_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}
      -DQ3X_GENERATOR=${CMAKE_GENERATOR}
      -DQ3X_CONFIG=$<CONFIG>
      -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/package_consumer_test.cmake
  )
endif()

install(TARGETS q3x_core q3x_text q3x_kernels q3x_engine
                qwen3x-orin qwen3x-inspect
  EXPORT Qwen3xOrinTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN "*.h"
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/generated/q3x/version.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/q3x
)
install(EXPORT Qwen3xOrinTargets
  FILE Qwen3xOrinTargets.cmake
  NAMESPACE q3x::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Qwen3xOrin
)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/generated/Qwen3xOrinConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/generated/Qwen3xOrinConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Qwen3xOrin
)
