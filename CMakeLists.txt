cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(Qwen3xOrin VERSION 0.7.0 LANGUAGES CXX CUDA)

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
option(Q3X_BUILD_GDN_C16_NORM_GATE_ADMISSION
  "Build the test-only exact-C16 GDN norm/gate admission route"
  OFF
)
option(Q3X_BUILD_GDN_CHUNK64_REFERENCE_ADMISSION
  "Build the test-only external-reference C64/WY GDN architecture route"
  OFF
)
option(Q3X_BUILD_GDN_CHUNK64_NATIVE_ADMISSION
  "Build the test-only native SM87 C64/WY GDN architecture route"
  OFF
)
option(Q3X_BUILD_GDN_PROMPT_WIDE_CHUNK_GRAPH_ADMISSION
  "Build the test-only P40 prompt-wide exact GDN chunk graph"
  OFF
)
option(Q3X_BUILD_NVFP4_MARLIN_PREFILL_ADMISSION
  "Build the test-only scheduler-wide vLLM-Marlin NVFP4 Prefill route"
  OFF
)
option(Q3X_BUILD_NVFP4_TRUE_LARGE_M_PREFILL_ADMISSION
  "Build the test-only WP-V2-C1 true-large-M NVFP4 Prefill route"
  OFF
)
option(Q3X_BUILD_NVFP4_G2_D2_PREFILL_ADMISSION
  "Build the test-only independent G2/D2 NVFP4 Prefill package"
  OFF
)
option(Q3X_BUILD_NVFP4_SHAPE_WIDE_PREFILL_ADMISSION
  "Build the test-only exact-P40000 shape-wide NVFP4 Prefill package"
  OFF
)
option(Q3X_BUILD_FP8_MARLIN_PREFILL_ADMISSION
  "Build the test-only frozen-vLLM FP8 W8A16 Marlin Prefill route"
  OFF
)
option(Q3X_BUILD_BF16_AB_LARGE_M_PREFILL_ADMISSION
  "Build the test-only Qwen3.6 BF16 A/B large-M Prefill route"
  OFF
)
option(Q3X_BUILD_FLASHINFER_PREFILL_ATTENTION_ADMISSION
  "Build the test-only FlashInfer SM87 Prefill-attention route"
  OFF
)
option(Q3X_BUILD_LAYER_WIDE_P40_MLP_ADMISSION
  "Build the test-only exact-P40000 layer-wide MLP plan"
  OFF
)
option(Q3X_BUILD_NVFP4_PERSISTENT_PREFILL_ADMISSION
  "Build the test-only exact-P40000 persistent NVFP4 MLP kernels"
  OFF
)
option(Q3X_BUILD_PROMPT_WIDE_P40_WHOLE_CORE_ADMISSION
  "Build the exact-P40000 whole-core Prefill admission substrate"
  OFF
)
option(Q3X_BUILD_P40_WHOLE_CORE_DEVELOPMENT_ROUTE
  "Build the default-off, accuracy-unqualified P40 whole-core development route"
  OFF
)
option(Q3X_BUILD_P40_PROJECTION_RESET_ADMISSION
  "Build the test-only exact-P40000 persistent S2R projection reset"
  OFF
)
option(Q3X_BUILD_P40_PHASE_BF16_PROJECTION_ADMISSION
  "Build the test-only rejected P40000 phase-local BF16 correctness primitives"
  OFF
)
option(Q3X_BUILD_P40_PACKED_PROJECTION_ADMISSION
  "Build the test-only AC-PREFILL-P40-PACKED-DATAFLOW-v1 projection route"
  OFF
)
option(Q3X_BUILD_P40_PACKED_NVFP4_V2_ADMISSION
  "Build the test-only shape-specific packed NVFP4 v2 execution route"
  OFF
)
option(Q3X_BUILD_P40_VLLM_MARLIN_PARITY_ADMISSION
  "Build the test-only exact-P40000 vLLM-Marlin projection host-dispatch reference"
  OFF
)
option(Q3X_BUILD_SM87_AOT_SYSTEM_V1_ADMISSION
  "Build the test-only host admission schema for AC-PREFILL-SM87-AOT-SYSTEM-v1"
  OFF
)
option(Q3X_BUILD_SM87_AOT_SYSTEM_V1_ARITHMETIC_WITNESS
  "Build the host-only real-P40 arithmetic-class witness for AC-PREFILL-SM87-AOT-SYSTEM-v1"
  OFF
)
option(Q3X_BUILD_SM87_TARGET_AOT_PROJECTION_V1_ADMISSION
  "Build the test-only byte-authenticated host transform, device-receipt schema, and compile-only NVFP4 CUDA feed for AC-PREFILL-SM87-AOT-SYSTEM-v1"
  OFF
)
option(Q3X_BUILD_SM87_TARGET_AOT_DEVICE_ASSETS_V1_ADMISSION
  "Build the test-only real-checkpoint target-AOT device owner, uploader, receipt issuer, and private ModelWeights attachment"
  OFF
)
option(Q3X_BUILD_SM87_TARGET_AOT_LAYER0_M192_ORACLE_ADMISSION
  "Build the test-only real-weight layer-0 M192 numerical/resource oracle for the private target-AOT NVFP4 kernels"
  OFF
)

# The owner-selected P40 baseline is one indivisible development inventory.
# Shadow the constituent cache options only in this top-level configure scope:
# the single bundle remains the user-facing control, while the individual
# test admissions remain OFF in the cache and cannot be enabled separately in
# a BUILD_TESTING=OFF build.  Do not FORCE-write the cache here.
if(Q3X_BUILD_P40_WHOLE_CORE_DEVELOPMENT_ROUTE)
  set(Q3X_BUILD_NVFP4_MARLIN_PREFILL_ADMISSION ON)
  set(Q3X_BUILD_FP8_MARLIN_PREFILL_ADMISSION ON)
  set(Q3X_BUILD_BF16_AB_LARGE_M_PREFILL_ADMISSION ON)
  set(Q3X_BUILD_FLASHINFER_PREFILL_ATTENTION_ADMISSION ON)
  set(Q3X_BUILD_GDN_CHUNK64_NATIVE_ADMISSION ON)
  set(Q3X_BUILD_GDN_PROMPT_WIDE_CHUNK_GRAPH_ADMISSION ON)
  set(Q3X_BUILD_LAYER_WIDE_P40_MLP_ADMISSION ON)
  set(Q3X_BUILD_NVFP4_PERSISTENT_PREFILL_ADMISSION ON)
  set(Q3X_BUILD_PROMPT_WIDE_P40_WHOLE_CORE_ADMISSION ON)
  message(STATUS
    "P40 whole-core v10 development inventory enabled (accuracy-unqualified; installation disabled)"
  )
endif()
if(Q3X_BUILD_GDN_B8_ADMISSION AND NOT BUILD_TESTING)
  message(FATAL_ERROR
    "Q3X_BUILD_GDN_B8_ADMISSION is test-only and requires BUILD_TESTING=ON"
  )
endif()
if(Q3X_BUILD_GDN_C16_NORM_GATE_ADMISSION AND NOT BUILD_TESTING)
  message(FATAL_ERROR
    "Q3X_BUILD_GDN_C16_NORM_GATE_ADMISSION is test-only and requires BUILD_TESTING=ON"
  )
endif()
if(Q3X_BUILD_GDN_CHUNK64_REFERENCE_ADMISSION AND NOT BUILD_TESTING)
  message(FATAL_ERROR
    "Q3X_BUILD_GDN_CHUNK64_REFERENCE_ADMISSION is test-only and requires BUILD_TESTING=ON"
  )
endif()
if(Q3X_BUILD_P40_WHOLE_CORE_DEVELOPMENT_ROUTE AND
   (NOT Q3X_BUILD_NVFP4_MARLIN_PREFILL_ADMISSION OR
    NOT Q3X_BUILD_FP8_MARLIN_PREFILL_ADMISSION OR
    NOT Q3X_BUILD_BF16_AB_LARGE_M_PREFILL_ADMISSION OR
    NOT Q3X_BUILD_FLASHINFER_PREFILL_ATTENTION_ADMISSION OR
    NOT Q3X_BUILD_GDN_CHUNK64_NATIVE_ADMISSION OR
    NOT Q3X_BUILD_GDN_PROMPT_WIDE_CHUNK_GRAPH_ADMISSION OR
    NOT Q3X_BUILD_LAYER_WIDE_P40_MLP_ADMISSION OR
    NOT Q3X_BUILD_NVFP4_PERSISTENT_PREFILL_ADMISSION OR
    NOT Q3X_BUILD_PROMPT_WIDE_P40_WHOLE_CORE_ADMISSION))
  message(FATAL_ERROR
    "Q3X_BUILD_P40_WHOLE_CORE_DEVELOPMENT_ROUTE requires the exact nine-item v10 whole-core inventory"
  )
endif()
if(Q3X_BUILD_GDN_CHUNK64_NATIVE_ADMISSION AND NOT BUILD_TESTING AND
   NOT Q3X_BUILD_P40_WHOLE_CORE_DEVELOPMENT_ROUTE)
  message(FATAL_ERROR
    "Q3X_BUILD_GDN_CHUNK64_NATIVE_ADMISSION is test-only and requires BUILD_TESTING=ON"
  )
endif()
if(Q3X_BUILD_GDN_PROMPT_WIDE_CHUNK_GRAPH_ADMISSION AND NOT BUILD_TESTING AND
   NOT Q3X_BUILD_P40_WHOLE_CORE_DEVELOPMENT_ROUTE)
  message(FATAL_ERROR
    "Q3X_BUILD_GDN_PROMPT_WIDE_CHUNK_GRAPH_ADMISSION is test-only and requires BUILD_TESTING=ON"
  )
endif()
if(Q3X_BUILD_GDN_PROMPT_WIDE_CHUNK_GRAPH_ADMISSION AND
   NOT Q3X_BUILD_GDN_CHUNK64_NATIVE_ADMISSION)
  message(FATAL_ERROR
    "Q3X_BUILD_GDN_PROMPT_WIDE_CHUNK_GRAPH_ADMISSION requires the native C64 GDN admission inventory"
  )
endif()
if(Q3X_BUILD_NVFP4_MARLIN_PREFILL_ADMISSION AND NOT BUILD_TESTING AND
   NOT Q3X_BUILD_P40_WHOLE_CORE_DEVELOPMENT_ROUTE)
  message(FATAL_ERROR
    "Q3X_BUILD_NVFP4_MARLIN_PREFILL_ADMISSION is test-only and requires BUILD_TESTING=ON"
  )
endif()
if(Q3X_BUILD_NVFP4_TRUE_LARGE_M_PREFILL_ADMISSION AND NOT BUILD_TESTING)
  message(FATAL_ERROR
    "Q3X_BUILD_NVFP4_TRUE_LARGE_M_PREFILL_ADMISSION is test-only and requires BUILD_TESTING=ON"
  )
endif()
if(Q3X_BUILD_NVFP4_TRUE_LARGE_M_PREFILL_ADMISSION AND
   (NOT Q3X_BUILD_NVFP4_MARLIN_PREFILL_ADMISSION OR
    NOT Q3X_BUILD_FP8_MARLIN_PREFILL_ADMISSION OR
    NOT Q3X_BUILD_GDN_CHUNK64_NATIVE_ADMISSION))
  message(FATAL_ERROR
    "Q3X_BUILD_NVFP4_TRUE_LARGE_M_PREFILL_ADMISSION requires the authenticated NVFP4/FP8 Marlin and native GDN admission inventories"
  )
endif()
if(Q3X_BUILD_NVFP4_G2_D2_PREFILL_ADMISSION AND NOT BUILD_TESTING)
  message(FATAL_ERROR
    "Q3X_BUILD_NVFP4_G2_D2_PREFILL_ADMISSION is test-only and requires BUILD_TESTING=ON"
  )
endif()
if(Q3X_BUILD_NVFP4_G2_D2_PREFILL_ADMISSION AND
   (NOT Q3X_BUILD_NVFP4_MARLIN_PREFILL_ADMISSION OR
    NOT Q3X_BUILD_FP8_MARLIN_PREFILL_ADMISSION OR
    NOT Q3X_BUILD_GDN_CHUNK64_NATIVE_ADMISSION))
  message(FATAL_ERROR
    "Q3X_BUILD_NVFP4_G2_D2_PREFILL_ADMISSION requires the authenticated NVFP4/FP8 Marlin and native GDN admission inventories"
  )
endif()
if(Q3X_BUILD_NVFP4_SHAPE_WIDE_PREFILL_ADMISSION AND NOT BUILD_TESTING)
  message(FATAL_ERROR
    "Q3X_BUILD_NVFP4_SHAPE_WIDE_PREFILL_ADMISSION is test-only and requires BUILD_TESTING=ON"
  )
endif()
if(Q3X_BUILD_FP8_MARLIN_PREFILL_ADMISSION AND NOT BUILD_TESTING AND
   NOT Q3X_BUILD_P40_WHOLE_CORE_DEVELOPMENT_ROUTE)
  message(FATAL_ERROR
    "Q3X_BUILD_FP8_MARLIN_PREFILL_ADMISSION is test-only and requires BUILD_TESTING=ON"
  )
endif()
if(Q3X_BUILD_BF16_AB_LARGE_M_PREFILL_ADMISSION AND NOT BUILD_TESTING AND
   NOT Q3X_BUILD_P40_WHOLE_CORE_DEVELOPMENT_ROUTE)
  message(FATAL_ERROR
    "Q3X_BUILD_BF16_AB_LARGE_M_PREFILL_ADMISSION is test-only and requires BUILD_TESTING=ON"
  )
endif()
if(Q3X_BUILD_FLASHINFER_PREFILL_ATTENTION_ADMISSION AND NOT BUILD_TESTING AND
   NOT Q3X_BUILD_P40_WHOLE_CORE_DEVELOPMENT_ROUTE)
  message(FATAL_ERROR
    "Q3X_BUILD_FLASHINFER_PREFILL_ATTENTION_ADMISSION is test-only and requires BUILD_TESTING=ON"
  )
endif()
if(Q3X_BUILD_LAYER_WIDE_P40_MLP_ADMISSION AND NOT BUILD_TESTING AND
   NOT Q3X_BUILD_P40_WHOLE_CORE_DEVELOPMENT_ROUTE)
  message(FATAL_ERROR
    "Q3X_BUILD_LAYER_WIDE_P40_MLP_ADMISSION is test-only and requires BUILD_TESTING=ON"
  )
endif()
if(Q3X_BUILD_NVFP4_PERSISTENT_PREFILL_ADMISSION AND NOT BUILD_TESTING AND
   NOT Q3X_BUILD_P40_WHOLE_CORE_DEVELOPMENT_ROUTE)
  message(FATAL_ERROR
    "Q3X_BUILD_NVFP4_PERSISTENT_PREFILL_ADMISSION is test-only and requires BUILD_TESTING=ON"
  )
endif()
if(Q3X_BUILD_NVFP4_PERSISTENT_PREFILL_ADMISSION AND
   (NOT Q3X_BUILD_NVFP4_MARLIN_PREFILL_ADMISSION OR
    NOT Q3X_BUILD_LAYER_WIDE_P40_MLP_ADMISSION))
  message(FATAL_ERROR
    "Q3X_BUILD_NVFP4_PERSISTENT_PREFILL_ADMISSION requires Marlin Prefill and layer-wide P40 admission inventories"
  )
endif()
if(Q3X_BUILD_PROMPT_WIDE_P40_WHOLE_CORE_ADMISSION AND NOT BUILD_TESTING AND
   NOT Q3X_BUILD_P40_WHOLE_CORE_DEVELOPMENT_ROUTE)
  message(FATAL_ERROR
    "Q3X_BUILD_PROMPT_WIDE_P40_WHOLE_CORE_ADMISSION is test-only and requires BUILD_TESTING=ON"
  )
endif()
if(Q3X_BUILD_PROMPT_WIDE_P40_WHOLE_CORE_ADMISSION AND
   (NOT Q3X_BUILD_NVFP4_MARLIN_PREFILL_ADMISSION OR
    NOT Q3X_BUILD_FP8_MARLIN_PREFILL_ADMISSION OR
    NOT Q3X_BUILD_BF16_AB_LARGE_M_PREFILL_ADMISSION OR
    NOT Q3X_BUILD_FLASHINFER_PREFILL_ATTENTION_ADMISSION OR
    NOT Q3X_BUILD_GDN_CHUNK64_NATIVE_ADMISSION OR
    NOT Q3X_BUILD_GDN_PROMPT_WIDE_CHUNK_GRAPH_ADMISSION OR
    NOT Q3X_BUILD_LAYER_WIDE_P40_MLP_ADMISSION OR
    NOT Q3X_BUILD_NVFP4_PERSISTENT_PREFILL_ADMISSION))
  message(FATAL_ERROR
    "Q3X_BUILD_PROMPT_WIDE_P40_WHOLE_CORE_ADMISSION requires the FP8/NVFP4 Marlin, BF16 A/B, FlashInfer, native prompt-wide GDN, layer-wide P40, and persistent NVFP4 admission inventories"
  )
endif()
if(Q3X_BUILD_P40_PROJECTION_RESET_ADMISSION AND NOT BUILD_TESTING)
  message(FATAL_ERROR
    "Q3X_BUILD_P40_PROJECTION_RESET_ADMISSION is test-only and requires BUILD_TESTING=ON"
  )
endif()
if(Q3X_BUILD_P40_PROJECTION_RESET_ADMISSION AND
   (NOT Q3X_BUILD_PROMPT_WIDE_P40_WHOLE_CORE_ADMISSION OR
    NOT Q3X_BUILD_NVFP4_MARLIN_PREFILL_ADMISSION OR
    NOT Q3X_BUILD_NVFP4_PERSISTENT_PREFILL_ADMISSION))
  message(FATAL_ERROR
    "Q3X_BUILD_P40_PROJECTION_RESET_ADMISSION requires the whole-core, NVFP4 Marlin, and interleaved persistent-sidecar inventories"
  )
endif()
if(Q3X_BUILD_P40_PHASE_BF16_PROJECTION_ADMISSION AND NOT BUILD_TESTING)
  message(FATAL_ERROR
    "Q3X_BUILD_P40_PHASE_BF16_PROJECTION_ADMISSION is test-only and requires BUILD_TESTING=ON"
  )
endif()
if(Q3X_BUILD_P40_PACKED_PROJECTION_ADMISSION AND NOT BUILD_TESTING)
  message(FATAL_ERROR
    "Q3X_BUILD_P40_PACKED_PROJECTION_ADMISSION is test-only and requires BUILD_TESTING=ON"
  )
endif()
if(Q3X_BUILD_P40_PACKED_PROJECTION_ADMISSION AND
   NOT Q3X_BUILD_PROMPT_WIDE_P40_WHOLE_CORE_ADMISSION)
  message(FATAL_ERROR
    "Q3X_BUILD_P40_PACKED_PROJECTION_ADMISSION requires the exact-P40000 whole-core execution substrate"
  )
endif()
if(Q3X_BUILD_P40_PACKED_NVFP4_V2_ADMISSION AND NOT BUILD_TESTING)
  message(FATAL_ERROR
    "Q3X_BUILD_P40_PACKED_NVFP4_V2_ADMISSION is test-only and requires BUILD_TESTING=ON"
  )
endif()
if(Q3X_BUILD_P40_PACKED_NVFP4_V2_ADMISSION AND
   (NOT Q3X_BUILD_P40_PACKED_PROJECTION_ADMISSION OR
    NOT Q3X_BUILD_PROMPT_WIDE_P40_WHOLE_CORE_ADMISSION))
  message(FATAL_ERROR
    "Q3X_BUILD_P40_PACKED_NVFP4_V2_ADMISSION requires the packed-v1 asset route and exact-P40000 whole-core substrate"
  )
endif()
if(Q3X_BUILD_P40_VLLM_MARLIN_PARITY_ADMISSION AND NOT BUILD_TESTING)
  message(FATAL_ERROR
    "Q3X_BUILD_P40_VLLM_MARLIN_PARITY_ADMISSION is test-only and requires BUILD_TESTING=ON"
  )
endif()
if(Q3X_BUILD_P40_VLLM_MARLIN_PARITY_ADMISSION AND
   (NOT Q3X_BUILD_PROMPT_WIDE_P40_WHOLE_CORE_ADMISSION OR
    NOT Q3X_BUILD_NVFP4_MARLIN_PREFILL_ADMISSION))
  message(FATAL_ERROR
    "Q3X_BUILD_P40_VLLM_MARLIN_PARITY_ADMISSION requires the exact-P40000 whole-core substrate and canonical NVFP4 Marlin inventory"
  )
endif()
if(Q3X_BUILD_SM87_AOT_SYSTEM_V1_ADMISSION AND NOT BUILD_TESTING)
  message(FATAL_ERROR
    "Q3X_BUILD_SM87_AOT_SYSTEM_V1_ADMISSION is a host-only test admission and requires BUILD_TESTING=ON"
  )
endif()
if(Q3X_BUILD_SM87_AOT_SYSTEM_V1_ARITHMETIC_WITNESS AND
   NOT BUILD_TESTING)
  message(FATAL_ERROR
    "Q3X_BUILD_SM87_AOT_SYSTEM_V1_ARITHMETIC_WITNESS is test-only and requires BUILD_TESTING=ON"
  )
endif()
if(Q3X_BUILD_SM87_AOT_SYSTEM_V1_ARITHMETIC_WITNESS AND
   NOT Q3X_BUILD_SM87_AOT_SYSTEM_V1_ADMISSION)
  message(FATAL_ERROR
    "Q3X_BUILD_SM87_AOT_SYSTEM_V1_ARITHMETIC_WITNESS requires the host-only SM87 AOT system-v1 schema admission"
  )
endif()
if(Q3X_BUILD_SM87_TARGET_AOT_PROJECTION_V1_ADMISSION AND NOT BUILD_TESTING)
  message(FATAL_ERROR
    "Q3X_BUILD_SM87_TARGET_AOT_PROJECTION_V1_ADMISSION is test-only and requires BUILD_TESTING=ON"
  )
endif()
if(Q3X_BUILD_SM87_TARGET_AOT_PROJECTION_V1_ADMISSION AND
   NOT Q3X_BUILD_SM87_AOT_SYSTEM_V1_ADMISSION)
  message(FATAL_ERROR
    "Q3X_BUILD_SM87_TARGET_AOT_PROJECTION_V1_ADMISSION requires the complete host-only SM87 AOT system contract"
  )
endif()
if(Q3X_BUILD_SM87_TARGET_AOT_DEVICE_ASSETS_V1_ADMISSION AND
   NOT BUILD_TESTING)
  message(FATAL_ERROR
    "Q3X_BUILD_SM87_TARGET_AOT_DEVICE_ASSETS_V1_ADMISSION is test-only and requires BUILD_TESTING=ON"
  )
endif()
if(Q3X_BUILD_SM87_TARGET_AOT_DEVICE_ASSETS_V1_ADMISSION AND
   NOT Q3X_BUILD_SM87_TARGET_AOT_PROJECTION_V1_ADMISSION)
  message(FATAL_ERROR
    "Q3X_BUILD_SM87_TARGET_AOT_DEVICE_ASSETS_V1_ADMISSION requires the byte-authenticated target-AOT projection admission"
  )
endif()
if(Q3X_BUILD_SM87_TARGET_AOT_LAYER0_M192_ORACLE_ADMISSION AND
   NOT BUILD_TESTING)
  message(FATAL_ERROR
    "Q3X_BUILD_SM87_TARGET_AOT_LAYER0_M192_ORACLE_ADMISSION is test-only and requires BUILD_TESTING=ON"
  )
endif()
if(Q3X_BUILD_SM87_TARGET_AOT_LAYER0_M192_ORACLE_ADMISSION AND
   NOT Q3X_BUILD_SM87_TARGET_AOT_DEVICE_ASSETS_V1_ADMISSION)
  message(FATAL_ERROR
    "Q3X_BUILD_SM87_TARGET_AOT_LAYER0_M192_ORACLE_ADMISSION requires the real-checkpoint target-AOT device assets admission"
  )
endif()
if((Q3X_BUILD_GDN_B8_ADMISSION AND
    Q3X_BUILD_GDN_C16_NORM_GATE_ADMISSION) OR
   (Q3X_BUILD_GDN_B8_ADMISSION AND
    Q3X_BUILD_GDN_CHUNK64_REFERENCE_ADMISSION) OR
   (Q3X_BUILD_GDN_B8_ADMISSION AND
    Q3X_BUILD_GDN_CHUNK64_NATIVE_ADMISSION) OR
   (Q3X_BUILD_GDN_C16_NORM_GATE_ADMISSION AND
    Q3X_BUILD_GDN_CHUNK64_REFERENCE_ADMISSION) OR
   (Q3X_BUILD_GDN_C16_NORM_GATE_ADMISSION AND
    Q3X_BUILD_GDN_CHUNK64_NATIVE_ADMISSION) OR
   (Q3X_BUILD_GDN_CHUNK64_REFERENCE_ADMISSION AND
    Q3X_BUILD_GDN_CHUNK64_NATIVE_ADMISSION))
  message(FATAL_ERROR
    "The GDN B8, C16 norm/gate, C64/WY reference, and C64/WY native admission routes are mutually exclusive"
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
  src/runtime/prefill_execution_plan.cpp
  src/runtime/prefill_operator_bindings.cpp
  src/runtime/prefill_workspace_plan.cpp
  src/runtime/prefill_route_evidence.cpp
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
if(Q3X_BUILD_LAYER_WIDE_P40_MLP_ADMISSION)
  # The plan-enabled query is an inline constexpr header contract. Propagate
  # the opt-in to every consumer so its definition is ODR-identical across the
  # planner, runner, engine, and host tests.
  target_compile_definitions(q3x_core PUBLIC
    Q3X_ENABLE_LAYER_WIDE_P40_MLP_ADMISSION=1
  )
endif()
if(Q3X_BUILD_PROMPT_WIDE_P40_WHOLE_CORE_ADMISSION)
  # The whole-core plan changes topology, request layout, runner scheduling,
  # and the sealed Engine authority. Keep one PUBLIC definition across every
  # consumer so default-off binaries cannot acquire a partial route.
  target_compile_definitions(q3x_core PUBLIC
    Q3X_ENABLE_PROMPT_WIDE_P40_WHOLE_CORE_ADMISSION=1
  )
endif()
if(Q3X_BUILD_P40_PROJECTION_RESET_ADMISSION)
  # The reset adds a distinct plan/tactic, runner schedule, engine authority,
  # and CUDA inventory.  Propagate one definition through every consumer so a
  # default-off binary cannot observe a partial route or an ODR-split enum.
  target_compile_definitions(q3x_core PUBLIC
    Q3X_ENABLE_P40_PROJECTION_RESET_ADMISSION=1
  )
endif()
if(Q3X_BUILD_P40_PACKED_PROJECTION_ADMISSION)
  # The packed route adds a distinct topology, sidecar ABI, runner path, and
  # sealed authority. Keep one definition across planner/runner/engine/tests.
  target_compile_definitions(q3x_core PUBLIC
    Q3X_ENABLE_P40_PACKED_PROJECTION_ADMISSION=1
  )
endif()
if(Q3X_BUILD_P40_PACKED_NVFP4_V2_ADMISSION)
  # V2 reuses the authenticated packed-v1 NVFP4 packing ABI through its own
  # NVFP4-only asset transaction, execution tactic, deployment identity, and
  # v14 witness. Keep admission consistent across planner/runner/engine/tests.
  target_compile_definitions(q3x_core PUBLIC
    Q3X_ENABLE_P40_PACKED_NVFP4_V2_ADMISSION=1
  )
endif()
if(Q3X_BUILD_P40_VLLM_MARLIN_PARITY_ADMISSION)
  # Keep the plan, workspace, runner, authority, and host witnesses on one
  # default-off admission bit.  The CUDA implementation has a separate
  # private definition below so default builds do not acquire the kernels.
  target_compile_definitions(q3x_core PUBLIC
    Q3X_ENABLE_P40_VLLM_MARLIN_PARITY_ADMISSION=1
  )
endif()
if(Q3X_BUILD_SM87_AOT_SYSTEM_V1_ADMISSION)
  # This bit exposes only a host descriptor schema. It has no CUDA sources,
  # launcher, selector, or production route, and has no dependency on any
  # rejected Prefill admission.
  target_compile_definitions(q3x_core PUBLIC
    Q3X_ENABLE_SM87_AOT_SYSTEM_V1_ADMISSION=1
  )
endif()
if(Q3X_BUILD_SM87_TARGET_AOT_PROJECTION_V1_ADMISSION)
  # This admission adds only a byte-authenticated host transformation,
  # device-receipt schema, and compile-only fail-closed CUDA feed. It does not
  # bind the runner or grant resource, numerical, performance, or production
  # authority.
  target_sources(q3x_core PRIVATE
    src/runtime/sm87_target_aot_projection_assets.cpp
  )
  target_compile_definitions(q3x_core PUBLIC
    Q3X_ENABLE_SM87_TARGET_AOT_PROJECTION_V1_ADMISSION=1
  )
endif()
target_sources(q3x_core PRIVATE
  src/runtime/sm87_target_aot_projection_device_assets.cpp
)
if(Q3X_BUILD_SM87_TARGET_AOT_DEVICE_ASSETS_V1_ADMISSION)
  # This bit adds only the engine-lifetime real-checkpoint arena owner,
  # bounded loader/uploader, device-readback receipt issuer, and private
  # ModelWeights lifetime attachment. It does not bind a runner route or open
  # the compile-only public launcher.
  target_compile_definitions(q3x_core PUBLIC
    Q3X_ENABLE_SM87_TARGET_AOT_DEVICE_ASSETS_V1_ADMISSION=1
  )
endif()
if(Q3X_BUILD_SM87_TARGET_AOT_LAYER0_M192_ORACLE_ADMISSION)
  # The source-private Oracle needs narrowly gated friendship in three
  # installed class definitions. Propagate one admission bit from the core
  # interface so core, kernels, engine, probes, tests, and package consumers
  # all preprocess those class definitions identically.
  target_compile_definitions(q3x_core PUBLIC
    Q3X_ENABLE_SM87_TARGET_AOT_LAYER0_M192_ORACLE_ADMISSION=1
  )
endif()
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
  src/runtime/reference_runner_decode_gqa_policy_internal.h
  src/runtime/reference_runner_full_attention_oracle_internal.h
  src/runtime/reference_runner_gdn_exact_span_policy_internal.h
  src/runtime/reference_runner_prompt_wide_policy_internal.h
  src/runtime/reference_runner_request_reset_policy_internal.h
  src/runtime/reference_runner_layer_major_views.cpp
  src/kernels/reference/decode_ops.cu
  src/kernels/reference/full_attention_c512_register_pipeline.cu
  src/kernels/reference/gdn_decode.cu
  src/kernels/reference/gdn_m16_register_state.cu
  src/kernels/reference/gdn_prefill_b8_sequential_sm87.cu
  src/kernels/reference/gdn_prefill_c16_norm_gate_sm87.cu
  src/kernels/reference/gdn_prefill_whole_span_conv_sm87.cu
  src/kernels/reference/gdn_prefill_whole_span_conv_sm87.h
  src/kernels/reference/layout_ops.cu
  src/kernels/reference/fp8_reference.cu
  src/kernels/reference/gemv_reference.cu
  src/kernels/reference/nvfp4_reference.cu
  src/kernels/reference/projection_dispatch.cpp
  src/kernels/sm87/device_probe.cu
  src/kernels/sm87/decode_gqa_splitkv_sm87.cu
  src/kernels/sm87/fp8_prefill_supermatrix.cu
  src/kernels/sm87/gdn_prefill_exact_span_sm87.cu
  src/kernels/sm87/gdn_prefill_exact_span_sm87.h
  src/kernels/sm87/nvfp4_marlin_w4a16.cu
  src/kernels/sm87/nvfp4_prefill_large_m.cu
  src/kernels/sm87/nvfp4_prefill_g2_d2.cu
  src/kernels/sm87/nvfp4_prefill_shape_wide.cu
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
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/runtime
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
if(Q3X_BUILD_GDN_C16_NORM_GATE_ADMISSION)
  target_compile_definitions(q3x_kernels PRIVATE
    Q3X_ENABLE_GDN_C16_NORM_GATE_ADMISSION=1
  )
endif()
if(Q3X_BUILD_GDN_CHUNK64_REFERENCE_ADMISSION OR
   Q3X_BUILD_GDN_CHUNK64_NATIVE_ADMISSION)
  target_sources(q3x_kernels PRIVATE
    src/kernels/reference/gdn_prefill_chunk64_cublas_reference_sm87.cu
    src/kernels/sm87/gdn_prefill_chunk_o_bv64_sm87.cu
    src/kernels/sm87/gdn_prefill_group_wy_sm87.cu
    src/kernels/sm87/gdn_prefill_wy_vllm_layout_sm87.cu
  )
endif()
if(Q3X_BUILD_GDN_CHUNK64_REFERENCE_ADMISSION)
  target_compile_definitions(q3x_kernels PRIVATE
    Q3X_ENABLE_GDN_CHUNK64_REFERENCE_ADMISSION=1
  )
endif()
if(Q3X_BUILD_NVFP4_MARLIN_PREFILL_ADMISSION)
  target_compile_definitions(q3x_kernels PRIVATE
    Q3X_ENABLE_NVFP4_MARLIN_PREFILL_ADMISSION=1
  )
endif()
if(Q3X_BUILD_NVFP4_TRUE_LARGE_M_PREFILL_ADMISSION)
  target_compile_definitions(q3x_kernels PRIVATE
    Q3X_ENABLE_NVFP4_TRUE_LARGE_M_PREFILL_ADMISSION=1
  )
endif()
if(Q3X_BUILD_NVFP4_G2_D2_PREFILL_ADMISSION)
  target_compile_definitions(q3x_kernels PRIVATE
    Q3X_ENABLE_NVFP4_G2_D2_PREFILL_ADMISSION=1
  )
endif()
if(Q3X_BUILD_NVFP4_SHAPE_WIDE_PREFILL_ADMISSION)
  target_compile_definitions(q3x_kernels PRIVATE
    Q3X_ENABLE_NVFP4_SHAPE_WIDE_PREFILL_ADMISSION=1
  )
endif()
if(Q3X_BUILD_NVFP4_PERSISTENT_PREFILL_ADMISSION)
  target_sources(q3x_kernels PRIVATE
    src/kernels/sm87/nvfp4_prefill_persistent.cu
  )
  target_compile_definitions(q3x_kernels PRIVATE
    Q3X_ENABLE_NVFP4_PERSISTENT_PREFILL_ADMISSION=1
  )
endif()
if(Q3X_BUILD_P40_PROJECTION_RESET_ADMISSION)
  target_sources(q3x_kernels PRIVATE
    src/kernels/sm87/p40_projection_reset.cu
  )
  target_compile_definitions(q3x_kernels PRIVATE
    Q3X_ENABLE_P40_PROJECTION_RESET_ADMISSION=1
  )
endif()
if(Q3X_BUILD_P40_PHASE_BF16_PROJECTION_ADMISSION)
  target_sources(q3x_kernels PRIVATE
    src/kernels/sm87/p40_phase_bf16_projection.cu
    src/kernels/sm87/phase_local_weight_expansion.cu
  )
  target_compile_definitions(q3x_kernels PRIVATE
    Q3X_ENABLE_P40_PHASE_BF16_PROJECTION_ADMISSION=1
  )
endif()
if(Q3X_BUILD_P40_PACKED_PROJECTION_ADMISSION)
  target_sources(q3x_kernels PRIVATE
    src/kernels/sm87/p40_packed_layout.cpp
    src/kernels/sm87/p40_packed_nvfp4.cu
    src/kernels/sm87/p40_packed_fp8.cu
  )
  target_compile_definitions(q3x_kernels PRIVATE
    Q3X_ENABLE_P40_PACKED_PROJECTION_ADMISSION=1
  )
endif()
if(Q3X_BUILD_P40_PACKED_NVFP4_V2_ADMISSION)
  target_sources(q3x_kernels PRIVATE
    src/kernels/sm87/p40_packed_nvfp4_v2.cu
  )
  target_compile_definitions(q3x_kernels PRIVATE
    Q3X_ENABLE_P40_PACKED_NVFP4_V2_ADMISSION=1
  )
endif()
if(Q3X_BUILD_P40_VLLM_MARLIN_PARITY_ADMISSION)
  target_sources(q3x_kernels PRIVATE
    src/kernels/sm87/nvfp4_marlin_p40_parity.cu
  )
  target_compile_definitions(q3x_kernels PRIVATE
    Q3X_ENABLE_NVFP4_MARLIN_P40_PARITY_ADMISSION=1
  )
endif()
if(Q3X_BUILD_SM87_TARGET_AOT_PROJECTION_V1_ADMISSION)
  target_sources(q3x_kernels PRIVATE
    src/kernels/sm87/sm87_target_aot_projection_nvfp4.cu
  )
  target_compile_definitions(q3x_kernels PRIVATE
    Q3X_ENABLE_SM87_TARGET_AOT_PROJECTION_V1_ADMISSION=1
  )
endif()
if(Q3X_BUILD_FP8_MARLIN_PREFILL_ADMISSION)
  target_sources(q3x_kernels PRIVATE
    src/kernels/sm87/fp8_marlin_w8a16.cu
  )
  target_compile_definitions(q3x_kernels PRIVATE
    Q3X_ENABLE_FP8_MARLIN_PREFILL_ADMISSION=1
  )
endif()
if(Q3X_BUILD_BF16_AB_LARGE_M_PREFILL_ADMISSION)
  target_sources(q3x_kernels PRIVATE
    src/kernels/sm87/bf16_ab_prefill.cu
  )
  target_compile_definitions(q3x_kernels PRIVATE
    Q3X_ENABLE_BF16_AB_LARGE_M_PREFILL_ADMISSION=1
  )
endif()
if(Q3X_BUILD_FLASHINFER_PREFILL_ATTENTION_ADMISSION)
  target_include_directories(q3x_kernels SYSTEM PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/flashinfer/include>
  )
  target_compile_definitions(q3x_kernels PRIVATE
    Q3X_ENABLE_FLASHINFER_PREFILL_ATTENTION_ADMISSION=1
  )
endif()
if(Q3X_BUILD_GDN_CHUNK64_NATIVE_ADMISSION)
  target_compile_definitions(q3x_kernels PRIVATE
    Q3X_ENABLE_GDN_CHUNK64_NATIVE_ADMISSION=1
  )
endif()
if(Q3X_BUILD_GDN_PROMPT_WIDE_CHUNK_GRAPH_ADMISSION)
  target_compile_definitions(q3x_kernels PRIVATE
    Q3X_ENABLE_GDN_PROMPT_WIDE_CHUNK_GRAPH_ADMISSION=1
  )
endif()
if(BUILD_TESTING)
  target_compile_definitions(q3x_kernels PRIVATE
    Q3X_ENABLE_REFERENCE_RUNNER_INTERNAL_TEST_SEAMS=1
  )
endif()
q3x_enable_warnings(q3x_kernels)

add_library(q3x_engine STATIC
  src/runtime/p40_packed_projection_assets.cpp
  src/runtime/reference_benchmark.cpp
  src/runtime/reference_engine.cpp
  src/runtime/reference_engine_control.cpp
  src/runtime/reference_engine_decode_sidecar_policy_internal.h
  src/runtime/reference_engine_final_token_policy_internal.h
  src/runtime/reference_engine_prefill_authority.cpp
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
if(Q3X_BUILD_NVFP4_MARLIN_PREFILL_ADMISSION)
  target_compile_definitions(q3x_engine PRIVATE
    Q3X_ENABLE_NVFP4_MARLIN_PREFILL_ADMISSION=1
  )
endif()
if(Q3X_BUILD_NVFP4_TRUE_LARGE_M_PREFILL_ADMISSION)
  target_compile_definitions(q3x_engine PRIVATE
    Q3X_ENABLE_NVFP4_TRUE_LARGE_M_PREFILL_ADMISSION=1
  )
endif()
if(Q3X_BUILD_NVFP4_G2_D2_PREFILL_ADMISSION)
  target_compile_definitions(q3x_engine PRIVATE
    Q3X_ENABLE_NVFP4_G2_D2_PREFILL_ADMISSION=1
  )
endif()
if(Q3X_BUILD_NVFP4_PERSISTENT_PREFILL_ADMISSION)
  target_compile_definitions(q3x_engine PRIVATE
    Q3X_ENABLE_NVFP4_PERSISTENT_PREFILL_ADMISSION=1
  )
endif()
if(Q3X_BUILD_FP8_MARLIN_PREFILL_ADMISSION)
  target_compile_definitions(q3x_engine PRIVATE
    Q3X_ENABLE_FP8_MARLIN_PREFILL_ADMISSION=1
  )
endif()
if(Q3X_BUILD_BF16_AB_LARGE_M_PREFILL_ADMISSION)
  target_compile_definitions(q3x_engine PRIVATE
    Q3X_ENABLE_BF16_AB_LARGE_M_PREFILL_ADMISSION=1
  )
endif()
if(Q3X_BUILD_FLASHINFER_PREFILL_ATTENTION_ADMISSION)
  target_compile_definitions(q3x_engine PRIVATE
    Q3X_ENABLE_FLASHINFER_PREFILL_ATTENTION_ADMISSION=1
  )
endif()
if(Q3X_BUILD_GDN_CHUNK64_NATIVE_ADMISSION)
  target_compile_definitions(q3x_engine PRIVATE
    Q3X_ENABLE_GDN_CHUNK64_NATIVE_ADMISSION=1
  )
endif()
if(Q3X_BUILD_GDN_PROMPT_WIDE_CHUNK_GRAPH_ADMISSION)
  target_compile_definitions(q3x_engine PRIVATE
    Q3X_ENABLE_GDN_PROMPT_WIDE_CHUNK_GRAPH_ADMISSION=1
  )
endif()
if(BUILD_TESTING)
  target_compile_definitions(q3x_engine PRIVATE
    Q3X_ENABLE_REFERENCE_ENGINE_INTERNAL_TEST_SEAMS=1
  )
endif()
q3x_enable_warnings(q3x_engine)

# Internal, deliberately narrow OpenAI/EvalScope evaluation adapter. It is
# kept separate from q3x_engine so the core runtime has no HTTP/protocol
# dependency and external evaluation cannot become a hidden production
# fallback path.
add_library(q3x_eval_gateway STATIC
  src/server/openai_protocol.cpp
  src/server/evaluation_server.cpp
)
target_compile_features(q3x_eval_gateway PUBLIC cxx_std_17)
target_include_directories(q3x_eval_gateway
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>
)
target_link_libraries(q3x_eval_gateway
  PUBLIC q3x::engine
  PRIVATE q3x::core Threads::Threads
)
if(BUILD_TESTING)
  target_compile_definitions(q3x_eval_gateway PRIVATE
    Q3X_EVAL_GATEWAY_BUILD_TESTING=1
  )
else()
  target_compile_definitions(q3x_eval_gateway PRIVATE
    Q3X_EVAL_GATEWAY_BUILD_TESTING=0
  )
endif()
if(Q3X_BUILD_P40_WHOLE_CORE_DEVELOPMENT_ROUTE)
  # This acknowledgement is deliberately narrower than the constituent
  # admission macros. It lets the evaluation gateway expose exactly the v10
  # development profile without making another experimental P40 successor or
  # an internal test seam reachable from a BUILD_TESTING=OFF binary.
  target_compile_definitions(q3x_eval_gateway PRIVATE
    Q3X_ENABLE_P40_WHOLE_CORE_DEVELOPMENT_ROUTE=1
  )
endif()
q3x_enable_warnings(q3x_eval_gateway)

add_executable(qwen3x-eval-server src/eval_server_main.cpp)
target_link_libraries(qwen3x-eval-server PRIVATE q3x_eval_gateway)
if(Q3X_BUILD_P40_WHOLE_CORE_DEVELOPMENT_ROUTE)
  target_compile_definitions(qwen3x-eval-server PRIVATE
    Q3X_ENABLE_P40_WHOLE_CORE_DEVELOPMENT_ROUTE=1
    Q3X_ENABLE_P40_WHOLE_CORE_TACTIC_CLI=1
  )
  # A development artifact must not be confused with the canonical adapter.
  set_target_properties(qwen3x-eval-server PROPERTIES
    OUTPUT_NAME qwen3x-eval-server-p40-v10-dev
  )
elseif(BUILD_TESTING AND Q3X_BUILD_PROMPT_WIDE_P40_WHOLE_CORE_ADMISSION)
  # Preserve the pre-existing test-only tactic surface without exposing the
  # typed development-route acknowledgement in ordinary test builds.
  target_compile_definitions(qwen3x-eval-server PRIVATE
    Q3X_ENABLE_P40_WHOLE_CORE_TACTIC_CLI=1
  )
endif()
q3x_enable_warnings(qwen3x-eval-server)

add_executable(qwen3x-orin src/main.cpp)
target_link_libraries(qwen3x-orin
  PRIVATE q3x::engine q3x::text q3x::core q3x::kernels
)
q3x_enable_warnings(qwen3x-orin)

add_executable(qwen3x-inspect tools/q3x-inspect/main.cpp)
target_link_libraries(qwen3x-inspect PRIVATE q3x::core)
q3x_enable_warnings(qwen3x-inspect)

if(BUILD_TESTING)
  find_package(Python3 QUIET COMPONENTS Interpreter)
  if(Python3_Interpreter_FOUND)
    add_test(
      NAME document_control
      COMMAND "${CMAKE_COMMAND}" -E env "PYTHONDONTWRITEBYTECODE=1"
              "${Python3_EXECUTABLE}"
              "${CMAKE_CURRENT_SOURCE_DIR}/tools/docs/validate_document_control.py"
              --repository "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    set_tests_properties(document_control PROPERTIES
      LABELS "documentation;host"
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    add_test(
      NAME document_control_unit
      COMMAND "${CMAKE_COMMAND}" -E env "PYTHONDONTWRITEBYTECODE=1"
              "${Python3_EXECUTABLE}" -m unittest tests.test_document_control
    )
    set_tests_properties(document_control_unit PROPERTIES
      LABELS "documentation;host"
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    add_test(
      NAME vllm_p40_geometry_host
      COMMAND "${CMAKE_COMMAND}" -E env "PYTHONDONTWRITEBYTECODE=1"
              "${Python3_EXECUTABLE}" -m unittest
              tests.test_vllm_p40_geometry
    )
    set_tests_properties(vllm_p40_geometry_host PROPERTIES
      LABELS "evaluation;host;prefill"
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    add_test(
      NAME evalscope_triplet_validator
      COMMAND "${CMAKE_COMMAND}" -E env "PYTHONDONTWRITEBYTECODE=1"
              "${Python3_EXECUTABLE}"
              "${CMAKE_CURRENT_SOURCE_DIR}/tests/evalscope_triplet_gate_test.py"
    )
    set_tests_properties(evalscope_triplet_validator PROPERTIES
      LABELS "evaluation;host"
    )
    add_test(
      NAME orin_perf_preflight_host
      COMMAND "${CMAKE_COMMAND}" -E env "PYTHONDONTWRITEBYTECODE=1"
              "${Python3_EXECUTABLE}"
              "${CMAKE_CURRENT_SOURCE_DIR}/tests/orin_perf_preflight_test.py"
    )
    set_tests_properties(orin_perf_preflight_host PROPERTIES
      LABELS "performance;preflight;host"
    )
    add_test(
      NAME target_aot_nvmap_recovery_host
      COMMAND "${CMAKE_COMMAND}" -E env "PYTHONDONTWRITEBYTECODE=1"
              "${Python3_EXECUTABLE}" -m unittest
              tests.test_classify_target_aot_nvmap_recovery
    )
    set_tests_properties(target_aot_nvmap_recovery_host PROPERTIES
      LABELS "correctness;checkpoint;host;sm87"
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    add_test(
      NAME target_aot_probe_lifecycle_host
      COMMAND "${CMAKE_COMMAND}" -E env "PYTHONDONTWRITEBYTECODE=1"
              "${Python3_EXECUTABLE}" -m unittest
              tests.test_run_target_aot_probe_with_nvmap
    )
    set_tests_properties(target_aot_probe_lifecycle_host PROPERTIES
      LABELS "correctness;checkpoint;host;sm87"
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
  endif()

  add_executable(q3x_reference_engine_final_token_policy_internal_test
    tests/reference_engine_final_token_policy_internal_test.cpp
  )
  target_include_directories(
    q3x_reference_engine_final_token_policy_internal_test
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/runtime"
  )
  q3x_enable_warnings(
    q3x_reference_engine_final_token_policy_internal_test
  )
  add_test(NAME reference_engine_final_token_policy_internal
    COMMAND q3x_reference_engine_final_token_policy_internal_test
  )
  set_tests_properties(
    reference_engine_final_token_policy_internal PROPERTIES
    LABELS "host;runtime;prefill;policy"
  )

  add_executable(q3x_reference_runner_prompt_wide_policy_internal_test
    tests/reference_runner_prompt_wide_policy_internal_test.cpp
  )
  target_include_directories(
    q3x_reference_runner_prompt_wide_policy_internal_test
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/runtime"
  )
  target_link_libraries(
    q3x_reference_runner_prompt_wide_policy_internal_test
    PRIVATE q3x::kernels
  )
  q3x_enable_warnings(
    q3x_reference_runner_prompt_wide_policy_internal_test
  )
  add_test(NAME reference_runner_prompt_wide_policy_internal
    COMMAND q3x_reference_runner_prompt_wide_policy_internal_test
  )
  set_tests_properties(
    reference_runner_prompt_wide_policy_internal PROPERTIES
    LABELS "host;runtime;prefill;policy"
  )

  add_executable(q3x_sm87_target_aot_oracle_admission_contract_test
    tests/sm87_target_aot_oracle_admission_contract_test.cpp
  )
  target_link_libraries(
    q3x_sm87_target_aot_oracle_admission_contract_test PRIVATE q3x::core
  )
  if(Q3X_BUILD_SM87_TARGET_AOT_LAYER0_M192_ORACLE_ADMISSION)
    target_compile_definitions(
      q3x_sm87_target_aot_oracle_admission_contract_test PRIVATE
      Q3X_EXPECT_SM87_TARGET_AOT_LAYER0_M192_ORACLE_ADMISSION=1
    )
  else()
    target_compile_definitions(
      q3x_sm87_target_aot_oracle_admission_contract_test PRIVATE
      Q3X_EXPECT_SM87_TARGET_AOT_LAYER0_M192_ORACLE_ADMISSION=0
    )
  endif()
  q3x_enable_warnings(q3x_sm87_target_aot_oracle_admission_contract_test)
  add_test(NAME sm87_target_aot_oracle_admission_contract
    COMMAND q3x_sm87_target_aot_oracle_admission_contract_test
  )
  set_tests_properties(sm87_target_aot_oracle_admission_contract PROPERTIES
    LABELS "host;sm87;prefill;target_aot;oracle;admission;odr"
  )

  add_test(NAME sm87_target_aot_oracle_header_surface_contract
    COMMAND "${CMAKE_COMMAND}"
      "-DQ3X_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
      "-DQ3X_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
      "-DQ3X_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}"
      "-DQ3X_CUDA_INCLUDE_DIRS=${CUDAToolkit_INCLUDE_DIRS}"
      "-DQ3X_EXPECT_ORACLE=${Q3X_BUILD_SM87_TARGET_AOT_LAYER0_M192_ORACLE_ADMISSION}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/VerifySm87TargetAotOracleHeaderSurface.cmake"
  )
  set_tests_properties(
    sm87_target_aot_oracle_header_surface_contract PROPERTIES
    LABELS "host;sm87;prefill;target_aot;oracle;installed_header;odr"
  )

  set(Q3X_E2E_MODEL_DIR "" CACHE PATH
      "Pinned Qwen3.6-27B-NVFP4 model directory for optional E2E tests")
  set(Q3X_REAL_PERF_MODEL_DIR "" CACHE PATH
      "Pinned Qwen3.6-27B-NVFP4 model directory for real-payload performance measurement")
  set(Q3X_REAL_PERF_CPU "11" CACHE STRING
      "Single host CPU used by serialized Orin performance measurement")
  option(Q3X_REQUIRE_REAL_PERF
         "Fail configuration or hardware checks instead of skipping real-payload measurement"
         OFF)
  set(_q3x_real_perf_model_dir "${Q3X_REAL_PERF_MODEL_DIR}")
  if(NOT _q3x_real_perf_model_dir AND
     DEFINED ENV{Q3X_REAL_PERF_MODEL_DIR})
    set(_q3x_real_perf_model_dir "$ENV{Q3X_REAL_PERF_MODEL_DIR}")
  endif()
  if(NOT _q3x_real_perf_model_dir AND Q3X_E2E_MODEL_DIR)
    set(_q3x_real_perf_model_dir "${Q3X_E2E_MODEL_DIR}")
  endif()
  if(NOT _q3x_real_perf_model_dir AND DEFINED ENV{Q3X_E2E_MODEL_DIR})
    set(_q3x_real_perf_model_dir "$ENV{Q3X_E2E_MODEL_DIR}")
  endif()
  if(Q3X_REQUIRE_REAL_PERF AND NOT _q3x_real_perf_model_dir)
    message(FATAL_ERROR
      "Q3X_REQUIRE_REAL_PERF=ON requires Q3X_REAL_PERF_MODEL_DIR or Q3X_E2E_MODEL_DIR")
  endif()
  if(NOT Q3X_REAL_PERF_CPU MATCHES "^[0-9]+$")
    message(FATAL_ERROR "Q3X_REAL_PERF_CPU must be one non-negative CPU id")
  endif()

  add_library(q3x_test_pinned_checkpoint STATIC
    tests/support/pinned_checkpoint.cpp
  )
  target_include_directories(q3x_test_pinned_checkpoint
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/tests/support"
  )
  target_link_libraries(q3x_test_pinned_checkpoint PUBLIC q3x::core)
  q3x_enable_warnings(q3x_test_pinned_checkpoint)

  # cuBLASLt is a benchmark-only external reference.  Keep its implementation
  # out of q3x_kernels and every installed/production target so the runtime has
  # no link-time route, fallback, or selection path into it.
  add_library(q3x_test_cublaslt_reference STATIC
    src/kernels/sm87/nvfp4_prefill_cublaslt.cu
  )
  target_compile_features(q3x_test_cublaslt_reference
    PUBLIC cxx_std_17 cuda_std_17
  )
  target_include_directories(q3x_test_cublaslt_reference
    PUBLIC
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/support"
    PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/include"
      "${CMAKE_CURRENT_BINARY_DIR}/generated"
  )
  target_link_libraries(q3x_test_cublaslt_reference
    PRIVATE CUDA::cudart CUDA::cublasLt
  )
  target_compile_options(q3x_test_cublaslt_reference PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  )
  q3x_enable_warnings(q3x_test_cublaslt_reference)

  add_executable(q3x_pinned_checkpoint_support_test
    tests/support/pinned_checkpoint_test.cpp
  )
  target_link_libraries(q3x_pinned_checkpoint_support_test
    PRIVATE q3x_test_pinned_checkpoint
  )
  q3x_enable_warnings(q3x_pinned_checkpoint_support_test)
  add_test(NAME pinned_checkpoint_support
    COMMAND q3x_pinned_checkpoint_support_test
  )
  set_tests_properties(pinned_checkpoint_support PROPERTIES
    LABELS "checkpoint;correctness;host"
  )
  if(_q3x_real_perf_model_dir)
    add_test(NAME pinned_checkpoint_support_real_model
      COMMAND q3x_pinned_checkpoint_support_test
              "${_q3x_real_perf_model_dir}"
    )
    set_tests_properties(pinned_checkpoint_support_real_model PROPERTIES
      LABELS "checkpoint;correctness;model"
      RUN_SERIAL TRUE
      TIMEOUT 120
    )
  endif()

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

  add_executable(q3x_sm87_nvfp4_prefill_large_m_host_test
    tests/sm87_nvfp4_prefill_large_m_host_test.cpp
  )
  target_link_libraries(q3x_sm87_nvfp4_prefill_large_m_host_test
    PRIVATE q3x::kernels
  )
  q3x_enable_warnings(q3x_sm87_nvfp4_prefill_large_m_host_test)
  add_test(NAME sm87_nvfp4_prefill_large_m_host
    COMMAND q3x_sm87_nvfp4_prefill_large_m_host_test
  )
  set_tests_properties(sm87_nvfp4_prefill_large_m_host PROPERTIES
    LABELS "host;sm87;prefill;contract"
  )

  add_executable(q3x_sm87_nvfp4_prefill_g2_d2_host_test
    tests/sm87_nvfp4_prefill_g2_d2_host_test.cpp
  )
  target_link_libraries(q3x_sm87_nvfp4_prefill_g2_d2_host_test
    PRIVATE q3x::kernels
  )
  q3x_enable_warnings(q3x_sm87_nvfp4_prefill_g2_d2_host_test)
  add_test(NAME sm87_nvfp4_prefill_g2_d2_host
    COMMAND q3x_sm87_nvfp4_prefill_g2_d2_host_test
  )
  set_tests_properties(sm87_nvfp4_prefill_g2_d2_host PROPERTIES
    LABELS "host;sm87;prefill;contract"
  )

  add_executable(q3x_sm87_nvfp4_prefill_shape_wide_host_test
    tests/sm87_nvfp4_prefill_shape_wide_host_test.cpp
  )
  target_link_libraries(q3x_sm87_nvfp4_prefill_shape_wide_host_test
    PRIVATE q3x::kernels
  )
  q3x_enable_warnings(q3x_sm87_nvfp4_prefill_shape_wide_host_test)
  add_test(NAME sm87_nvfp4_prefill_shape_wide_host
    COMMAND q3x_sm87_nvfp4_prefill_shape_wide_host_test
  )
  set_tests_properties(sm87_nvfp4_prefill_shape_wide_host PROPERTIES
    LABELS "host;sm87;prefill;contract"
  )

  if(Q3X_BUILD_NVFP4_PERSISTENT_PREFILL_ADMISSION)
    add_executable(q3x_sm87_nvfp4_prefill_persistent_host_test
      tests/sm87_nvfp4_prefill_persistent_host_test.cpp
    )
    target_link_libraries(q3x_sm87_nvfp4_prefill_persistent_host_test
      PRIVATE q3x::kernels
    )
    q3x_enable_warnings(q3x_sm87_nvfp4_prefill_persistent_host_test)
    add_test(NAME sm87_nvfp4_prefill_persistent_host
      COMMAND q3x_sm87_nvfp4_prefill_persistent_host_test
    )
    set_tests_properties(sm87_nvfp4_prefill_persistent_host PROPERTIES
      LABELS "host;sm87;prefill;contract"
    )
  endif()

  if(Q3X_BUILD_P40_PROJECTION_RESET_ADMISSION)
    add_executable(q3x_sm87_p40_projection_reset_host_test
      tests/sm87_p40_projection_reset_host_test.cpp
    )
    target_link_libraries(q3x_sm87_p40_projection_reset_host_test
      PRIVATE q3x::kernels
    )
    q3x_enable_warnings(q3x_sm87_p40_projection_reset_host_test)
    add_test(NAME sm87_p40_projection_reset_host
      COMMAND q3x_sm87_p40_projection_reset_host_test
    )
    set_tests_properties(sm87_p40_projection_reset_host PROPERTIES
      LABELS "host;sm87;prefill;contract"
    )
  endif()

  if(Q3X_BUILD_P40_PACKED_PROJECTION_ADMISSION)
    add_executable(q3x_sm87_p40_packed_projection_host_test
      tests/sm87_p40_packed_projection_host_test.cpp
    )
    target_link_libraries(q3x_sm87_p40_packed_projection_host_test
      PRIVATE q3x::kernels
    )
    q3x_enable_warnings(q3x_sm87_p40_packed_projection_host_test)
    add_test(NAME sm87_p40_packed_projection_host
      COMMAND q3x_sm87_p40_packed_projection_host_test
    )
    set_tests_properties(sm87_p40_packed_projection_host PROPERTIES
      LABELS "host;sm87;prefill;contract;packed"
    )

    add_executable(q3x_sm87_p40_packed_fp8_cuda_test
      tests/sm87_p40_packed_fp8_cuda_test.cu
    )
    target_link_libraries(q3x_sm87_p40_packed_fp8_cuda_test
      PRIVATE q3x::kernels CUDA::cudart
    )
    q3x_enable_warnings(q3x_sm87_p40_packed_fp8_cuda_test)
    add_test(NAME sm87_p40_packed_fp8_cuda
      COMMAND q3x_sm87_p40_packed_fp8_cuda_test
    )
    set_tests_properties(sm87_p40_packed_fp8_cuda PROPERTIES
      LABELS "cuda;sm87;prefill;correctness;packed"
      SKIP_RETURN_CODE 77
      RESOURCE_LOCK q3x_gpu
    )

    add_executable(q3x_sm87_p40_packed_nvfp4_cuda_test
      tests/sm87_p40_packed_nvfp4_cuda_test.cu
    )
    target_link_libraries(q3x_sm87_p40_packed_nvfp4_cuda_test
      PRIVATE q3x::kernels CUDA::cudart
    )
    q3x_enable_warnings(q3x_sm87_p40_packed_nvfp4_cuda_test)
    add_test(NAME sm87_p40_packed_nvfp4_cuda
      COMMAND q3x_sm87_p40_packed_nvfp4_cuda_test
    )
    set_tests_properties(sm87_p40_packed_nvfp4_cuda PROPERTIES
      LABELS "cuda;sm87;prefill;correctness;packed"
      SKIP_RETURN_CODE 77
      RESOURCE_LOCK q3x_gpu
    )
  endif()

  if(Q3X_BUILD_P40_PACKED_NVFP4_V2_ADMISSION)
    add_executable(q3x_sm87_p40_packed_nvfp4_v2_cuda_test
      tests/sm87_p40_packed_nvfp4_v2_cuda_test.cu
    )
    target_link_libraries(q3x_sm87_p40_packed_nvfp4_v2_cuda_test
      PRIVATE q3x::kernels CUDA::cudart
    )
    q3x_enable_warnings(q3x_sm87_p40_packed_nvfp4_v2_cuda_test)
    add_test(NAME sm87_p40_packed_nvfp4_v2_cuda
      COMMAND q3x_sm87_p40_packed_nvfp4_v2_cuda_test
    )
    set_tests_properties(sm87_p40_packed_nvfp4_v2_cuda PROPERTIES
      LABELS "cuda;sm87;prefill;correctness;packed;v2"
      SKIP_RETURN_CODE 77
      RESOURCE_LOCK q3x_gpu
    )
  endif()

  if(Q3X_BUILD_P40_VLLM_MARLIN_PARITY_ADMISSION)
    add_executable(q3x_sm87_nvfp4_marlin_p40_parity_host_test
      tests/sm87_nvfp4_marlin_p40_parity_host_test.cpp
    )
    target_link_libraries(q3x_sm87_nvfp4_marlin_p40_parity_host_test
      PRIVATE q3x::kernels
    )
    q3x_enable_warnings(q3x_sm87_nvfp4_marlin_p40_parity_host_test)
    add_test(NAME sm87_nvfp4_marlin_p40_parity_host
      COMMAND q3x_sm87_nvfp4_marlin_p40_parity_host_test
    )
    set_tests_properties(sm87_nvfp4_marlin_p40_parity_host PROPERTIES
      LABELS "host;sm87;prefill;contract;parity;default_off"
    )

    add_executable(q3x_sm87_nvfp4_marlin_p40_parity_cuda_test
      tests/sm87_nvfp4_marlin_p40_parity_cuda_test.cu
    )
    target_link_libraries(q3x_sm87_nvfp4_marlin_p40_parity_cuda_test
      PRIVATE q3x::kernels CUDA::cudart
    )
    q3x_enable_warnings(q3x_sm87_nvfp4_marlin_p40_parity_cuda_test)
    add_test(NAME sm87_nvfp4_marlin_p40_parity_cuda
      COMMAND q3x_sm87_nvfp4_marlin_p40_parity_cuda_test
    )
    set_tests_properties(sm87_nvfp4_marlin_p40_parity_cuda PROPERTIES
      LABELS "cuda;sm87;prefill;topology;parity;default_off"
      SKIP_RETURN_CODE 77
      RESOURCE_LOCK q3x_gpu
    )
  endif()

  if(Q3X_BUILD_P40_PHASE_BF16_PROJECTION_ADMISSION)
    add_executable(q3x_sm87_p40_phase_bf16_projection_host_test
      tests/sm87_p40_phase_bf16_projection_host_test.cpp
    )
    target_link_libraries(q3x_sm87_p40_phase_bf16_projection_host_test
      PRIVATE q3x::kernels
    )
    q3x_enable_warnings(q3x_sm87_p40_phase_bf16_projection_host_test)
    add_test(NAME sm87_p40_phase_bf16_projection_host
      COMMAND q3x_sm87_p40_phase_bf16_projection_host_test
    )
    set_tests_properties(sm87_p40_phase_bf16_projection_host PROPERTIES
      LABELS "host;sm87;prefill;projection;correctness;rejected;default_off"
    )

    add_executable(q3x_sm87_phase_local_weight_expansion_host_test
      tests/sm87_phase_local_weight_expansion_host_test.cpp
    )
    target_link_libraries(q3x_sm87_phase_local_weight_expansion_host_test
      PRIVATE q3x::kernels
    )
    q3x_enable_warnings(q3x_sm87_phase_local_weight_expansion_host_test)
    add_test(NAME sm87_phase_local_weight_expansion_host
      COMMAND q3x_sm87_phase_local_weight_expansion_host_test
    )
    set_tests_properties(sm87_phase_local_weight_expansion_host PROPERTIES
      LABELS "host;sm87;prefill;projection;quantization;correctness;rejected;default_off"
    )

    add_executable(q3x_sm87_phase_local_weight_expansion_cuda_test
      tests/sm87_phase_local_weight_expansion_cuda_test.cu
    )
    target_link_libraries(q3x_sm87_phase_local_weight_expansion_cuda_test
      PRIVATE q3x::kernels CUDA::cudart
    )
    q3x_enable_warnings(q3x_sm87_phase_local_weight_expansion_cuda_test)
    add_test(NAME sm87_phase_local_weight_expansion_cuda
      COMMAND q3x_sm87_phase_local_weight_expansion_cuda_test
    )
    set_tests_properties(sm87_phase_local_weight_expansion_cuda PROPERTIES
      LABELS "cuda;sm87;prefill;projection;quantization;correctness;synthetic;rejected;default_off"
      SKIP_RETURN_CODE 77
      RESOURCE_LOCK q3x_sm87_gpu
    )

    add_executable(q3x_sm87_p40_phase_bf16_projection_cuda_test
      tests/sm87_p40_phase_bf16_projection_cuda_test.cu
    )
    target_link_libraries(q3x_sm87_p40_phase_bf16_projection_cuda_test
      PRIVATE q3x::kernels CUDA::cudart
    )
    q3x_enable_warnings(q3x_sm87_p40_phase_bf16_projection_cuda_test)
    add_test(NAME sm87_p40_phase_bf16_projection_cuda
      COMMAND q3x_sm87_p40_phase_bf16_projection_cuda_test
    )
    set_tests_properties(sm87_p40_phase_bf16_projection_cuda PROPERTIES
      LABELS "cuda;sm87;prefill;projection;correctness;synthetic;rejected;default_off"
      SKIP_RETURN_CODE 77
      RESOURCE_LOCK q3x_sm87_gpu
    )
  endif()

  if(Q3X_BUILD_BF16_AB_LARGE_M_PREFILL_ADMISSION)
    add_executable(q3x_sm87_bf16_ab_prompt_wide_p40_host_test
      tests/sm87_bf16_ab_prompt_wide_p40_host_test.cpp
    )
    target_link_libraries(q3x_sm87_bf16_ab_prompt_wide_p40_host_test
      PRIVATE q3x::kernels
    )
    q3x_enable_warnings(q3x_sm87_bf16_ab_prompt_wide_p40_host_test)
    add_test(NAME sm87_bf16_ab_prompt_wide_p40_host
      COMMAND q3x_sm87_bf16_ab_prompt_wide_p40_host_test
    )
    set_tests_properties(sm87_bf16_ab_prompt_wide_p40_host PROPERTIES
      LABELS "host;sm87;prefill;contract"
    )

    add_executable(q3x_sm87_bf16_ab_prompt_wide_p40_cuda_test
      tests/sm87_bf16_ab_prompt_wide_p40_cuda_test.cu
    )
    target_link_libraries(q3x_sm87_bf16_ab_prompt_wide_p40_cuda_test
      PRIVATE q3x::kernels
    )
    q3x_enable_warnings(q3x_sm87_bf16_ab_prompt_wide_p40_cuda_test)
    add_test(NAME sm87_bf16_ab_prompt_wide_p40_cuda
      COMMAND q3x_sm87_bf16_ab_prompt_wide_p40_cuda_test
    )
    set_tests_properties(sm87_bf16_ab_prompt_wide_p40_cuda PROPERTIES
      LABELS "cuda;sm87;prefill;correctness;synthetic;graph"
      SKIP_RETURN_CODE 77
      RESOURCE_LOCK q3x_sm87_gpu
    )
  endif()

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

  add_executable(q3x_sm87_nvfp4_marlin_layer0_test
    tests/sm87_nvfp4_marlin_layer0_test.cu
  )
  target_link_libraries(q3x_sm87_nvfp4_marlin_layer0_test
    PRIVATE q3x::kernels q3x_test_pinned_checkpoint CUDA::cudart
  )
  q3x_enable_warnings(q3x_sm87_nvfp4_marlin_layer0_test)
  add_test(NAME sm87_nvfp4_marlin_layer0
    COMMAND q3x_sm87_nvfp4_marlin_layer0_test
            "${_q3x_real_perf_model_dir}"
  )
  add_test(NAME sm87_nvfp4_marlin_layer0_m407
    COMMAND q3x_sm87_nvfp4_marlin_layer0_test
            "${_q3x_real_perf_model_dir}" --tokens=407
  )
  add_test(NAME sm87_nvfp4_marlin_layer0_m481
    COMMAND q3x_sm87_nvfp4_marlin_layer0_test
            "${_q3x_real_perf_model_dir}" --tokens=481
  )
  set_tests_properties(sm87_nvfp4_marlin_layer0
                       sm87_nvfp4_marlin_layer0_m407
                       sm87_nvfp4_marlin_layer0_m481 PROPERTIES
    LABELS "cuda;sm87;prefill;correctness;checkpoint"
    SKIP_RETURN_CODE 77
    RESOURCE_LOCK q3x_sm87_gpu
  )

  if(Q3X_BUILD_FP8_MARLIN_PREFILL_ADMISSION)
    # Compile/link sentinel for the direct frozen-vLLM W8A16 port.  It invokes
    # only pre-device argument rejection, so this first admission gate requires
    # no GPU and does not claim real-payload correctness or performance.
    add_executable(q3x_sm87_fp8_marlin_layer0_compile_test
      tests/sm87_fp8_marlin_layer0_compile_test.cu
    )
    target_link_libraries(q3x_sm87_fp8_marlin_layer0_compile_test
      PRIVATE q3x::kernels CUDA::cudart
    )
    q3x_enable_warnings(q3x_sm87_fp8_marlin_layer0_compile_test)
    add_test(NAME sm87_fp8_marlin_layer0_compile
      COMMAND q3x_sm87_fp8_marlin_layer0_compile_test
    )
    set_tests_properties(sm87_fp8_marlin_layer0_compile PROPERTIES
      LABELS "host;sm87;prefill;compile_contract"
    )

    add_executable(q3x_sm87_fp8_marlin_real_qkv_test
      tests/sm87_fp8_marlin_real_qkv_test.cu
    )
    target_link_libraries(q3x_sm87_fp8_marlin_real_qkv_test
      PRIVATE q3x::kernels q3x_test_pinned_checkpoint CUDA::cudart
    )
    q3x_enable_warnings(q3x_sm87_fp8_marlin_real_qkv_test)
    add_test(NAME sm87_fp8_marlin_real_qkv_m407
      COMMAND q3x_sm87_fp8_marlin_real_qkv_test
              "${_q3x_real_perf_model_dir}"
    )
    set_tests_properties(sm87_fp8_marlin_real_qkv_m407 PROPERTIES
      LABELS "cuda;sm87;prefill;correctness;checkpoint;performance"
      SKIP_RETURN_CODE 77
      RESOURCE_LOCK q3x_sm87_gpu
    )
  endif()

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

  # Test-only zero-workspace cuBLASLt integration screen for exact C512 Gate.
  # It compares canonical NVFP4 direct dequantization + BF16 cuBLASLt against
  # the public production M128 dispatcher without altering production code.
  add_executable(q3x_nvfp4_prefill_gate_c512_cublaslt_ceiling
    tests/sm87_nvfp4_prefill_gate_c512_cublaslt_ceiling.cu
  )
  target_link_libraries(q3x_nvfp4_prefill_gate_c512_cublaslt_ceiling
    PRIVATE q3x::kernels CUDA::cublasLt
  )
  q3x_enable_warnings(q3x_nvfp4_prefill_gate_c512_cublaslt_ceiling)
  add_test(NAME nvfp4_prefill_gate_c512_cublaslt_ceiling
    COMMAND q3x_nvfp4_prefill_gate_c512_cublaslt_ceiling --mode=smoke
  )
  set_tests_properties(nvfp4_prefill_gate_c512_cublaslt_ceiling PROPERTIES
    LABELS "cuda;sm87;prefill;correctness;synthetic;smoke"
    SKIP_RETURN_CODE 77
  )

  # Test-only exact-C512 Down P0 ceiling. It keeps the production runner and
  # public integration untouched while comparing the current production M128
  # kernel with canonical NVFP4 direct dequantization + zero-workspace Lt.
  add_executable(q3x_nvfp4_prefill_down_c512_cublaslt_ceiling
    tests/sm87_nvfp4_prefill_down_c512_cublaslt_ceiling.cu
  )
  target_link_libraries(q3x_nvfp4_prefill_down_c512_cublaslt_ceiling
    PRIVATE q3x::kernels q3x_test_pinned_checkpoint
            q3x_test_cublaslt_reference CUDA::cublasLt
  )
  q3x_enable_warnings(q3x_nvfp4_prefill_down_c512_cublaslt_ceiling)
  add_test(NAME nvfp4_prefill_down_c512_cublaslt_ceiling
    COMMAND q3x_nvfp4_prefill_down_c512_cublaslt_ceiling --mode=smoke
  )
  set_tests_properties(nvfp4_prefill_down_c512_cublaslt_ceiling PROPERTIES
    LABELS "cuda;sm87;prefill;correctness;synthetic;smoke"
    SKIP_RETURN_CODE 77
  )

  # Benchmark-only exact-C512 NVFP4 direct-dequant + zero-workspace cuBLASLt
  # reference contract. This independently locks scalar decode equivalence,
  # native-M128 bitwise output, two-node graph capture, bounds, and fail-closed
  # zero-enqueue validation without exposing a production route.
  add_executable(q3x_sm87_nvfp4_prefill_cublaslt_module_test
    tests/sm87_nvfp4_prefill_cublaslt_module_test.cu
  )
  target_link_libraries(q3x_sm87_nvfp4_prefill_cublaslt_module_test
    PRIVATE q3x::kernels q3x_test_cublaslt_reference
  )
  q3x_enable_warnings(q3x_sm87_nvfp4_prefill_cublaslt_module_test)
  add_test(NAME sm87_nvfp4_prefill_cublaslt_module
    COMMAND q3x_sm87_nvfp4_prefill_cublaslt_module_test
  )
  set_tests_properties(sm87_nvfp4_prefill_cublaslt_module PROPERTIES
    LABELS "cuda;sm87;prefill;module"
    SKIP_RETURN_CODE 77
  )

  # Independent external-reference test for exact-C512 Down Window8
  # direct-dequant + zero-workspace cuBLASLt.  This target is test-only and
  # has no runner ownership, dispatch, fallback, or promotion eligibility.
  add_executable(q3x_sm87_nvfp4_prefill_down_cublaslt_module_test
    tests/sm87_nvfp4_prefill_down_cublaslt_module_test.cu
  )
  target_link_libraries(q3x_sm87_nvfp4_prefill_down_cublaslt_module_test
    PRIVATE q3x::kernels q3x_test_cublaslt_reference
  )
  q3x_enable_warnings(q3x_sm87_nvfp4_prefill_down_cublaslt_module_test)
  add_test(NAME sm87_nvfp4_prefill_down_cublaslt_module
    COMMAND q3x_sm87_nvfp4_prefill_down_cublaslt_module_test
  )
  set_tests_properties(sm87_nvfp4_prefill_down_cublaslt_module PROPERTIES
    LABELS "cuda;sm87;prefill;module;down"
    SKIP_RETURN_CODE 77
  )

  # Isolated exact-C512 Gate/Up external-reference and synthetic smoke screen.
  # This is the only pair-screen target that imports cuBLASLt.
  add_executable(q3x_nvfp4_prefill_gate_up_c512_cublaslt_pair_screen
    tests/sm87_nvfp4_prefill_gate_up_c512_cublaslt_pair_screen.cu
  )
  target_link_libraries(
    q3x_nvfp4_prefill_gate_up_c512_cublaslt_pair_screen
    PRIVATE q3x::kernels q3x_test_pinned_checkpoint CUDA::cublasLt CUDA::cudart
  )
  q3x_enable_warnings(
    q3x_nvfp4_prefill_gate_up_c512_cublaslt_pair_screen
  )
  add_test(NAME nvfp4_prefill_gate_up_c512_cublaslt_pair_screen
    COMMAND q3x_nvfp4_prefill_gate_up_c512_cublaslt_pair_screen --mode=smoke
  )
  set_tests_properties(
    nvfp4_prefill_gate_up_c512_cublaslt_pair_screen PROPERTIES
    LABELS "cuda;sm87;prefill;correctness;synthetic;smoke"
    SKIP_RETURN_CODE 77
  )

  # The real-weight native retention executable is compiled from the shared
  # harness with the external-reference code physically absent.  It has no Lt
  # header, symbol, dynamic dependency, or BF16 weight scratch allocation.
  add_executable(q3x_nvfp4_prefill_gate_up_c512_native_retention
    tests/sm87_nvfp4_prefill_gate_up_c512_cublaslt_pair_screen.cu
  )
  target_compile_definitions(
    q3x_nvfp4_prefill_gate_up_c512_native_retention
    PRIVATE Q3X_NATIVE_RETENTION_ONLY=1
  )
  target_link_libraries(
    q3x_nvfp4_prefill_gate_up_c512_native_retention
    PRIVATE q3x::kernels q3x_test_pinned_checkpoint CUDA::cudart
  )
  q3x_enable_warnings(q3x_nvfp4_prefill_gate_up_c512_native_retention)
  add_test(NAME nvfp4_prefill_gate_up_c512_native_only_smoke
    COMMAND q3x_nvfp4_prefill_gate_up_c512_native_retention --mode=smoke
  )
  set_tests_properties(
    nvfp4_prefill_gate_up_c512_native_only_smoke PROPERTIES
    LABELS "cuda;sm87;prefill;correctness;synthetic;smoke;native_only"
    SKIP_RETURN_CODE 77
  )

  set(_q3x_cublaslt_audit_prefix
      "${CMAKE_CURRENT_BINARY_DIR}/cublaslt-isolation-install")
  add_test(NAME production_cublaslt_isolation
    COMMAND "${CMAKE_COMMAND}"
      "-DQ3X_BUILD_DIR=${CMAKE_CURRENT_BINARY_DIR}"
      "-DQ3X_AUDIT_PREFIX=${_q3x_cublaslt_audit_prefix}"
      "-DQ3X_PRODUCTION_EXECUTABLE=$<TARGET_FILE:qwen3x-orin>"
      "-DQ3X_KERNEL_ARCHIVE=$<TARGET_FILE:q3x_kernels>"
      "-DQ3X_ENGINE_ARCHIVE=$<TARGET_FILE:q3x_engine>"
      "-DQ3X_NATIVE_RETENTION_EXECUTABLE=$<TARGET_FILE:q3x_nvfp4_prefill_gate_up_c512_native_retention>"
      "-DQ3X_INSTALLED_EXECUTABLE=${_q3x_cublaslt_audit_prefix}/${CMAKE_INSTALL_BINDIR}/qwen3x-orin"
      "-DQ3X_INSTALLED_KERNEL_ARCHIVE=${_q3x_cublaslt_audit_prefix}/${CMAKE_INSTALL_LIBDIR}/${CMAKE_STATIC_LIBRARY_PREFIX}q3x_kernels${CMAKE_STATIC_LIBRARY_SUFFIX}"
      "-DQ3X_INSTALLED_ENGINE_ARCHIVE=${_q3x_cublaslt_audit_prefix}/${CMAKE_INSTALL_LIBDIR}/${CMAKE_STATIC_LIBRARY_PREFIX}q3x_engine${CMAKE_STATIC_LIBRARY_SUFFIX}"
      "-DQ3X_NM=${CMAKE_NM}"
      "-DQ3X_READELF=${CMAKE_READELF}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/VerifyCublasLtIsolation.cmake"
  )
  set_tests_properties(production_cublaslt_isolation PROPERTIES
    LABELS "host;policy;production_isolation"
    RUN_SERIAL TRUE
  )

  if(_q3x_real_perf_model_dir)
    find_program(_q3x_taskset_command NAMES taskset)
    if(NOT _q3x_taskset_command)
      message(FATAL_ERROR
        "Real-payload performance measurement requires the taskset command")
    endif()
    add_test(NAME nvfp4_prefill_down_c512_checkpoint_reference
      COMMAND "${_q3x_taskset_command}" --cpu-list "${Q3X_REAL_PERF_CPU}"
              $<TARGET_FILE:q3x_nvfp4_prefill_down_c512_cublaslt_ceiling>
              --mode=performance-checkpoint
              --checkpoint=${_q3x_real_perf_model_dir}
    )
    add_test(NAME nvfp4_prefill_gate_up_c512_checkpoint_retention
      COMMAND "${_q3x_taskset_command}" --cpu-list "${Q3X_REAL_PERF_CPU}"
              $<TARGET_FILE:q3x_nvfp4_prefill_gate_up_c512_native_retention>
              --mode=performance-checkpoint
              --native=m64n256bs512
              --checkpoint=${_q3x_real_perf_model_dir}
    )
    set_tests_properties(
      nvfp4_prefill_down_c512_checkpoint_reference
      PROPERTIES
      LABELS "cuda;sm87;prefill;performance;checkpoint;external_reference"
      RUN_SERIAL TRUE
      RESOURCE_LOCK q3x_sm87_gpu
      TIMEOUT 900
    )
    set_tests_properties(
      nvfp4_prefill_gate_up_c512_checkpoint_retention
      PROPERTIES
      LABELS "cuda;sm87;prefill;performance;checkpoint;retention"
      RUN_SERIAL TRUE
      RESOURCE_LOCK q3x_sm87_gpu
      TIMEOUT 900
    )
    if(NOT Q3X_REQUIRE_REAL_PERF)
      set_tests_properties(
        nvfp4_prefill_down_c512_checkpoint_reference
        nvfp4_prefill_gate_up_c512_checkpoint_retention
        PROPERTIES SKIP_RETURN_CODE 77
      )
    endif()
  endif()

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

  # Strictly test-only Z C512 canonical raw-B XOR+U16 register-feed cell.
  # This reuses the two-variant screen harness without adding a persistent
  # sidecar or changing production dispatch and ModelWeights.
  add_executable(q3x_fp8_prefill_z_m128_canonical_xor_register_feed_screen
    tests/sm87_fp8_prefill_qkv_m128_split_m64_screen.cu
  )
  target_compile_definitions(
    q3x_fp8_prefill_z_m128_canonical_xor_register_feed_screen
    PRIVATE Q3X_FP8_Z_CANONICAL_XOR_SCREEN=1
  )
  target_link_libraries(
    q3x_fp8_prefill_z_m128_canonical_xor_register_feed_screen
    PRIVATE q3x::kernels
  )
  q3x_enable_warnings(
    q3x_fp8_prefill_z_m128_canonical_xor_register_feed_screen
  )
  add_test(NAME fp8_prefill_z_m128_canonical_xor_register_feed_validate
    COMMAND q3x_fp8_prefill_z_m128_canonical_xor_register_feed_screen
            --mode=validate
  )
  set_tests_properties(
    fp8_prefill_z_m128_canonical_xor_register_feed_validate PROPERTIES
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

  # Focused synthetic correctness/guard contract for the production
  # exact-C512 FP8 projection-supermatrix route. End-to-end performance stays
  # a real-checkpoint/EvalScope responsibility outside this target.
  add_executable(q3x_sm87_fp8_prefill_supermatrix_v2_test
    tests/sm87_fp8_prefill_supermatrix_v2_test.cu
  )
  target_link_libraries(q3x_sm87_fp8_prefill_supermatrix_v2_test
    PRIVATE q3x::kernels CUDA::cudart
  )
  q3x_enable_warnings(q3x_sm87_fp8_prefill_supermatrix_v2_test)
  add_test(NAME sm87_fp8_prefill_supermatrix_v2
    COMMAND q3x_sm87_fp8_prefill_supermatrix_v2_test
  )
  set_tests_properties(sm87_fp8_prefill_supermatrix_v2 PROPERTIES
    LABELS "cuda;sm87;prefill;correctness;synthetic"
    SKIP_RETURN_CODE 77
    RESOURCE_LOCK q3x_sm87_gpu
    TIMEOUT 180
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

  add_executable(q3x_full_attention_group_q64_panel_cuda_test
    tests/full_attention_group_q64_panel_cuda_test.cu
  )
  target_link_libraries(q3x_full_attention_group_q64_panel_cuda_test
    PRIVATE q3x::core q3x::kernels
  )
  target_compile_options(q3x_full_attention_group_q64_panel_cuda_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  )
  q3x_enable_warnings(q3x_full_attention_group_q64_panel_cuda_test)
  add_test(NAME full_attention_group_q64_panel_cuda
    COMMAND q3x_full_attention_group_q64_panel_cuda_test
  )
  set_tests_properties(full_attention_group_q64_panel_cuda PROPERTIES
    LABELS "cuda;sm87;prefill;attention;architecture_candidate"
    RESOURCE_LOCK q3x_sm87_gpu
    RUN_SERIAL TRUE
    TIMEOUT 300
  )

  # Default-off Attention-v4 bring-up.  The target exercises only the
  # Explicit Q128-v4 correctness surface. Runtime dispatch remains confined
  # to the default-off, accuracy-unqualified layer-major tactic.
  add_executable(q3x_full_attention_group_q128_v4_cuda_test
    tests/full_attention_group_q128_v4_cuda_test.cu
  )
  target_link_libraries(q3x_full_attention_group_q128_v4_cuda_test
    PRIVATE q3x::core q3x::kernels
  )
  target_compile_options(q3x_full_attention_group_q128_v4_cuda_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  )
  q3x_enable_warnings(q3x_full_attention_group_q128_v4_cuda_test)
  add_test(NAME full_attention_group_q128_v4_cuda
    COMMAND q3x_full_attention_group_q128_v4_cuda_test
  )
  set_tests_properties(full_attention_group_q128_v4_cuda PROPERTIES
    LABELS "cuda;sm87;prefill;attention;local_mutation;dependency_only;default_off"
    RESOURCE_LOCK q3x_sm87_gpu
    RUN_SERIAL TRUE
    TIMEOUT 300
  )

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

  # Isolated exact-C16 GDN plus plain-RMSNorm/SiLU epilogue screen. The
  # production implementation is linked from q3x_kernels; this translation
  # unit compiles only the standalone correctness/resource harness.
  add_executable(q3x_gdn_prefill_c16_norm_gate_sm87_candidate
    tests/gdn_prefill_c16_norm_gate_sm87_candidate.cu
    src/kernels/reference/gdn_prefill_c16_norm_gate_sm87.cu
  )
  target_link_libraries(q3x_gdn_prefill_c16_norm_gate_sm87_candidate
    PRIVATE q3x::core q3x::kernels
  )
  target_compile_options(q3x_gdn_prefill_c16_norm_gate_sm87_candidate PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-Xptxas=-v>
  )
  target_compile_definitions(q3x_gdn_prefill_c16_norm_gate_sm87_candidate
    PRIVATE
      Q3X_GDN_C16_NORM_GATE_STANDALONE=1
      Q3X_GDN_C16_NORM_GATE_USE_LINKED_IMPLEMENTATION=1
  )
  q3x_enable_warnings(q3x_gdn_prefill_c16_norm_gate_sm87_candidate)
  add_test(NAME gdn_prefill_c16_norm_gate_sm87_candidate
    COMMAND q3x_gdn_prefill_c16_norm_gate_sm87_candidate
  )
  set_tests_properties(gdn_prefill_c16_norm_gate_sm87_candidate PROPERTIES
    LABELS "correctness;cuda;prefill;sm87;synthetic"
    SKIP_RETURN_CODE 77
  )

  # Production exact whole-span GDN recurrence. This compares C256/C512
  # output, recurrent state, convolution history, replay, guards, and resource
  # bounds against the retained exact-C16 chain used by the fallback path.
  add_executable(q3x_gdn_prefill_exact_span_component_test
    tests/gdn_prefill_exact_span_component_test.cu
  )
  target_include_directories(q3x_gdn_prefill_exact_span_component_test
    PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/sm87"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/reference"
  )
  target_link_libraries(q3x_gdn_prefill_exact_span_component_test
    PRIVATE q3x::kernels CUDA::cudart
  )
  q3x_enable_warnings(q3x_gdn_prefill_exact_span_component_test)
  add_test(NAME gdn_prefill_exact_span_component
    COMMAND q3x_gdn_prefill_exact_span_component_test
  )
  set_tests_properties(gdn_prefill_exact_span_component PROPERTIES
    LABELS "correctness;cuda;prefill;sm87;synthetic"
    SKIP_RETURN_CODE 77
    RESOURCE_LOCK q3x_sm87_gpu
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
  target_include_directories(q3x_reference_runner_host_test
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/runtime"
  )
  target_link_libraries(q3x_reference_runner_host_test PRIVATE q3x::kernels)
  q3x_enable_warnings(q3x_reference_runner_host_test)
  add_test(NAME reference_runner_host COMMAND q3x_reference_runner_host_test)

  add_executable(q3x_reference_runner_layer_major_views_host_test
    tests/reference_runner_layer_major_views_host_test.cpp
  )
  target_link_libraries(q3x_reference_runner_layer_major_views_host_test
    PRIVATE q3x::kernels
  )
  q3x_enable_warnings(q3x_reference_runner_layer_major_views_host_test)
  add_test(NAME reference_runner_layer_major_views_host
    COMMAND q3x_reference_runner_layer_major_views_host_test
  )

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

  add_executable(q3x_reference_engine_decode_sidecar_policy_host_test
    tests/reference_engine_decode_sidecar_policy_host_test.cpp
  )
  target_include_directories(
    q3x_reference_engine_decode_sidecar_policy_host_test
    PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/include"
      "${CMAKE_CURRENT_BINARY_DIR}/generated"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/runtime"
  )
  q3x_enable_warnings(q3x_reference_engine_decode_sidecar_policy_host_test)
  add_test(NAME reference_engine_decode_sidecar_policy_host
    COMMAND q3x_reference_engine_decode_sidecar_policy_host_test
  )
  set_tests_properties(reference_engine_decode_sidecar_policy_host PROPERTIES
    LABELS "host;engine;decode;production;authority"
  )

  add_executable(q3x_bound_prefill_authority_type_test
    tests/bound_prefill_authority_type_test.cpp
  )
  target_include_directories(q3x_bound_prefill_authority_type_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/runtime"
  )
  target_link_libraries(q3x_bound_prefill_authority_type_test
    PRIVATE q3x::engine
  )
  q3x_enable_warnings(q3x_bound_prefill_authority_type_test)
  add_test(NAME bound_prefill_authority_type
    COMMAND q3x_bound_prefill_authority_type_test
  )
  set_tests_properties(bound_prefill_authority_type PROPERTIES
    LABELS "host;engine;prefill;authority"
  )

  add_executable(q3x_openai_protocol_test
    tests/openai_protocol_test.cpp
  )
  target_link_libraries(q3x_openai_protocol_test
    PRIVATE q3x_eval_gateway q3x::core
  )
  q3x_enable_warnings(q3x_openai_protocol_test)
  add_test(NAME openai_protocol
    COMMAND q3x_openai_protocol_test
            "${CMAKE_CURRENT_SOURCE_DIR}/.q3x-work/tests/openai-protocol"
  )
  add_test(NAME eval_server_help COMMAND qwen3x-eval-server --help)
  if(Q3X_BUILD_P40_WHOLE_CORE_DEVELOPMENT_ROUTE)
    set_tests_properties(eval_server_help PROPERTIES
      PASS_REGULAR_EXPRESSION "development-route p40-whole-core-v10"
    )
    add_test(NAME eval_server_p40_v10_dev_help
      COMMAND qwen3x-eval-server --help
    )
    set_tests_properties(eval_server_p40_v10_dev_help PROPERTIES
      PASS_REGULAR_EXPRESSION "development-route p40-whole-core-v10"
    )
    add_test(NAME eval_server_p40_v10_rejects_profile_override_after
      COMMAND qwen3x-eval-server unused
              --development-route p40-whole-core-v10
              --max-sequence-length 40001
    )
    add_test(NAME eval_server_p40_v10_rejects_profile_override_before
      COMMAND qwen3x-eval-server unused
              --max-sequence-length 40001
              --development-route p40-whole-core-v10
    )
    add_test(NAME eval_server_p40_v10_rejects_unacknowledged_tactic
      COMMAND qwen3x-eval-server unused
              --prefill-execution-mode layer-major
              --prefill-attention-tactic
              native-flashinfer-exact-whole-prompt
              --prefill-projection-tactic
              native-prompt-wide-p40-whole-core
    )
    add_test(NAME eval_server_p40_v10_rejects_missing_selector
      COMMAND qwen3x-eval-server unused
    )
    add_test(NAME eval_server_p40_v10_rejects_route_environment
      COMMAND ${CMAKE_COMMAND} -E env
              Q3X_RUN_DECODE_GQA_SPLITKV_ADMISSION=1
              $<TARGET_FILE:qwen3x-eval-server> unused
              --development-route p40-whole-core-v10
    )
    set_tests_properties(
      eval_server_p40_v10_rejects_profile_override_after
      eval_server_p40_v10_rejects_profile_override_before
      eval_server_p40_v10_rejects_unacknowledged_tactic
      eval_server_p40_v10_rejects_missing_selector
      eval_server_p40_v10_rejects_route_environment
      PROPERTIES WILL_FAIL TRUE LABELS "host;server;prefill;p40;development"
    )
  else()
    set_tests_properties(eval_server_help PROPERTIES
      PASS_REGULAR_EXPRESSION
        "q3x.sm87.production.p40.legacy-c512-exact.v3"
    )
    add_test(NAME eval_server_rejects_unbuilt_p40_v10_development_route
      COMMAND qwen3x-eval-server unused
              --development-route p40-whole-core-v10
    )
    add_test(NAME eval_server_rejects_unbuilt_p40_v10_projection_tactic
      COMMAND qwen3x-eval-server unused
              --prefill-projection-tactic
              native-prompt-wide-p40-whole-core
    )
    add_test(NAME eval_server_rejects_unbuilt_p40_v10_attention_tactic
      COMMAND qwen3x-eval-server unused
              --prefill-attention-tactic
              native-flashinfer-exact-whole-prompt
    )
    set_tests_properties(
      eval_server_rejects_unbuilt_p40_v10_development_route
      eval_server_rejects_unbuilt_p40_v10_projection_tactic
      eval_server_rejects_unbuilt_p40_v10_attention_tactic
      PROPERTIES WILL_FAIL TRUE LABELS "host;server;prefill;p40;isolation"
    )
  endif()
  if(NOT Q3X_BUILD_P40_WHOLE_CORE_DEVELOPMENT_ROUTE)
    add_test(NAME eval_server_production_help_contract
      COMMAND ${CMAKE_COMMAND}
        -DQ3X_EVAL_SERVER=$<TARGET_FILE:qwen3x-eval-server>
        -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/eval_server_production_help_test.cmake
    )
    set_tests_properties(eval_server_production_help_contract PROPERTIES
      LABELS "host;server;production;authority"
    )
    add_test(NAME eval_server_nonloopback_requires_api_key
      COMMAND qwen3x-eval-server unused --host 0.0.0.0
    )
    set_tests_properties(eval_server_nonloopback_requires_api_key PROPERTIES
      WILL_FAIL TRUE
      PASS_REGULAR_EXPRESSION
        "a non-loopback bind requires --api-key-file"
      LABELS "host;server;production;authentication"
    )
    add_test(NAME eval_server_rejects_duplicate_api_key_file
      COMMAND qwen3x-eval-server unused
              --api-key-file first.key --api-key-file second.key
    )
    set_tests_properties(eval_server_rejects_duplicate_api_key_file PROPERTIES
      WILL_FAIL TRUE
      PASS_REGULAR_EXPRESSION
        "--api-key-file may be specified only once"
      LABELS "host;server;production;authentication"
    )
  endif()
  add_test(NAME eval_server_rejects_invalid_prefill_mode
    COMMAND qwen3x-eval-server unused
            --prefill-execution-mode invalid
  )
  set_tests_properties(eval_server_rejects_invalid_prefill_mode PROPERTIES
    WILL_FAIL TRUE
  )
  add_test(NAME eval_server_rejects_unsealed_layer_major_route
    COMMAND qwen3x-eval-server unused
            --prefill-execution-mode layer-major
            --projection-backend reference
  )
  set_tests_properties(
    eval_server_rejects_unsealed_layer_major_route PROPERTIES
    WILL_FAIL TRUE
  )
  add_test(NAME eval_server_rejects_segmented_projection_in_legacy_mode
    COMMAND qwen3x-eval-server unused
            --prefill-projection-tactic segmented-marlin-operator-panel
  )
  set_tests_properties(
    eval_server_rejects_segmented_projection_in_legacy_mode PROPERTIES
    WILL_FAIL TRUE
  )
  add_test(NAME eval_server_rejects_native_large_m_projection_in_legacy_mode
    COMMAND qwen3x-eval-server unused
            --prefill-projection-tactic
            native-quantized-large-m-operator-panel
  )
  set_tests_properties(
    eval_server_rejects_native_large_m_projection_in_legacy_mode PROPERTIES
    WILL_FAIL TRUE
  )
  add_test(NAME eval_server_rejects_true_large_m_nvfp4_in_legacy_mode
    COMMAND qwen3x-eval-server unused
            --prefill-projection-tactic
            native-nvfp4-true-large-m-operator-panel
  )
  set_tests_properties(
    eval_server_rejects_true_large_m_nvfp4_in_legacy_mode PROPERTIES
    WILL_FAIL TRUE
  )
  add_test(NAME eval_server_rejects_nvfp4_g2_d2_in_legacy_mode
    COMMAND qwen3x-eval-server unused
            --prefill-projection-tactic
            native-nvfp4-g2-d2-large-m-operator-panel
  )
  set_tests_properties(
    eval_server_rejects_nvfp4_g2_d2_in_legacy_mode PROPERTIES
    WILL_FAIL TRUE
  )
  add_test(NAME eval_server_rejects_q128_attention_in_legacy_mode
    COMMAND qwen3x-eval-server unused
            --prefill-attention-tactic native-group-q128-v4-panel
  )
  set_tests_properties(
    eval_server_rejects_q128_attention_in_legacy_mode PROPERTIES
    WILL_FAIL TRUE
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

  if(Q3X_BUILD_NVFP4_MARLIN_PREFILL_ADMISSION AND
     Q3X_BUILD_FP8_MARLIN_PREFILL_ADMISSION AND
     Q3X_BUILD_GDN_CHUNK64_NATIVE_ADMISSION)
    add_executable(
      q3x_reference_prefill_operator_panel_equivalence_e2e_test
      tests/reference_prefill_operator_panel_equivalence_e2e_test.cpp
    )
    target_include_directories(
      q3x_reference_prefill_operator_panel_equivalence_e2e_test
      PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/runtime"
    )
    target_link_libraries(
      q3x_reference_prefill_operator_panel_equivalence_e2e_test
      PRIVATE q3x::engine q3x::core
    )
    q3x_enable_warnings(
      q3x_reference_prefill_operator_panel_equivalence_e2e_test
    )
    add_test(NAME reference_prefill_operator_panel_equivalence_e2e
      COMMAND q3x_reference_prefill_operator_panel_equivalence_e2e_test
              "${_q3x_e2e_model_dir}"
    )
    set_tests_properties(
      reference_prefill_operator_panel_equivalence_e2e PROPERTIES
      LABELS "e2e;model;sm87;prefill;operator_panel;correctness;bitwise"
      RESOURCE_LOCK q3x_sm87_gpu
      RUN_SERIAL TRUE
      SKIP_RETURN_CODE 77
      TIMEOUT 3600
    )
    if(Q3X_BUILD_FLASHINFER_PREFILL_ATTENTION_ADMISSION)
      add_test(NAME reference_prefill_flashinfer_exact_panel_e2e
        COMMAND q3x_reference_prefill_operator_panel_equivalence_e2e_test
                "${_q3x_e2e_model_dir}"
      )
      set_tests_properties(
        reference_prefill_flashinfer_exact_panel_e2e PROPERTIES
        LABELS "e2e;model;sm87;prefill;operator_panel;architecture_candidate;accuracy_unqualified"
        ENVIRONMENT
          "Q3X_TEST_PREFILL_ATTENTION_TACTIC=native-flashinfer-exact-panel"
        RESOURCE_LOCK q3x_sm87_gpu
        RUN_SERIAL TRUE
        SKIP_RETURN_CODE 77
        TIMEOUT 3600
      )
    endif()
  endif()

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

  add_executable(q3x_reference_prefill_all_prompt_tokens_admission_e2e_test
    tests/reference_prefill_all_prompt_tokens_admission_e2e_test.cpp
  )
  target_include_directories(
    q3x_reference_prefill_all_prompt_tokens_admission_e2e_test
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/runtime"
  )
  target_link_libraries(
    q3x_reference_prefill_all_prompt_tokens_admission_e2e_test
    PRIVATE q3x::engine
  )
  q3x_enable_warnings(
    q3x_reference_prefill_all_prompt_tokens_admission_e2e_test
  )
  add_test(NAME reference_prefill_all_prompt_tokens_admission_e2e
    COMMAND q3x_reference_prefill_all_prompt_tokens_admission_e2e_test
            "${_q3x_e2e_model_dir}"
  )
  set_tests_properties(
    reference_prefill_all_prompt_tokens_admission_e2e PROPERTIES
    LABELS "e2e;model;sm87;prefill;whole_prompt_admission"
    RUN_SERIAL TRUE
    SKIP_RETURN_CODE 77
    TIMEOUT 3600
  )

  add_executable(q3x_reference_runner_prompt_wide_policy_e2e_test
    tests/reference_runner_prompt_wide_policy_e2e_test.cpp
  )
  target_include_directories(
    q3x_reference_runner_prompt_wide_policy_e2e_test
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/runtime"
  )
  target_link_libraries(
    q3x_reference_runner_prompt_wide_policy_e2e_test
    PRIVATE q3x::engine q3x::core
  )
  q3x_enable_warnings(q3x_reference_runner_prompt_wide_policy_e2e_test)
  add_test(NAME reference_runner_prompt_wide_policy_e2e
    COMMAND q3x_reference_runner_prompt_wide_policy_e2e_test
            "${_q3x_e2e_model_dir}"
  )
  set_tests_properties(
    reference_runner_prompt_wide_policy_e2e PROPERTIES
    LABELS "e2e;model;sm87;prefill;prompt_wide;correctness;bitwise"
    RESOURCE_LOCK q3x_sm87_gpu
    RUN_SERIAL TRUE
    SKIP_RETURN_CODE 77
    TIMEOUT 3600
  )

  add_executable(q3x_reference_prefill_single_arbitrary_tile_admission_e2e_test
    tests/reference_prefill_single_arbitrary_tile_admission_e2e_test.cpp
  )
  target_include_directories(
    q3x_reference_prefill_single_arbitrary_tile_admission_e2e_test
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/runtime"
  )
  target_link_libraries(
    q3x_reference_prefill_single_arbitrary_tile_admission_e2e_test
    PRIVATE q3x::engine
  )
  q3x_enable_warnings(
    q3x_reference_prefill_single_arbitrary_tile_admission_e2e_test
  )
  add_test(NAME reference_prefill_single_arbitrary_tile_admission_e2e
    COMMAND q3x_reference_prefill_single_arbitrary_tile_admission_e2e_test
            "${_q3x_e2e_model_dir}"
  )
  set_tests_properties(
    reference_prefill_single_arbitrary_tile_admission_e2e PROPERTIES
    LABELS "e2e;model;sm87;prefill;single_arbitrary_tile_admission"
    RUN_SERIAL TRUE
    SKIP_RETURN_CODE 77
    TIMEOUT 3600
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

  if(Q3X_BUILD_GDN_C16_NORM_GATE_ADMISSION)
    add_executable(q3x_reference_gdn_prefill_c16_norm_gate_engine_e2e_test
      tests/reference_gdn_prefill_c16_norm_gate_engine_e2e_test.cpp
    )
    target_include_directories(
      q3x_reference_gdn_prefill_c16_norm_gate_engine_e2e_test
      PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/runtime"
    )
    target_link_libraries(
      q3x_reference_gdn_prefill_c16_norm_gate_engine_e2e_test
      PRIVATE q3x::engine
    )
    q3x_enable_warnings(
      q3x_reference_gdn_prefill_c16_norm_gate_engine_e2e_test
    )
    add_test(NAME reference_gdn_prefill_c16_norm_gate_engine_e2e
      COMMAND q3x_reference_gdn_prefill_c16_norm_gate_engine_e2e_test
              "${_q3x_e2e_model_dir}"
    )
    set_tests_properties(
      reference_gdn_prefill_c16_norm_gate_engine_e2e PROPERTIES
      LABELS "e2e;model;sm87;prefill;gdn;admission;correctness;ability"
      RUN_SERIAL TRUE
      SKIP_RETURN_CODE 77
      TIMEOUT 1800
    )
  endif()

  if(Q3X_BUILD_GDN_CHUNK64_REFERENCE_ADMISSION)
    add_executable(
      q3x_reference_gdn_prefill_chunk64_reference_engine_e2e_test
      tests/reference_gdn_prefill_chunk64_reference_engine_e2e_test.cpp
    )
    target_include_directories(
      q3x_reference_gdn_prefill_chunk64_reference_engine_e2e_test
      PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/runtime"
    )
    target_link_libraries(
      q3x_reference_gdn_prefill_chunk64_reference_engine_e2e_test
      PRIVATE q3x::engine
    )
    q3x_enable_warnings(
      q3x_reference_gdn_prefill_chunk64_reference_engine_e2e_test
    )
    add_test(NAME reference_gdn_prefill_chunk64_reference_engine_e2e
      COMMAND q3x_reference_gdn_prefill_chunk64_reference_engine_e2e_test
              "${_q3x_e2e_model_dir}"
    )
    set_tests_properties(
      reference_gdn_prefill_chunk64_reference_engine_e2e PROPERTIES
      LABELS "e2e;model;sm87;prefill;gdn;architecture;reference"
      RUN_SERIAL TRUE
      SKIP_RETURN_CODE 77
      TIMEOUT 1800
    )
  endif()

  if(Q3X_BUILD_GDN_CHUNK64_NATIVE_ADMISSION)
    add_executable(
      q3x_gdn_prefill_chunk_o_bv64_component_test
      tests/gdn_prefill_chunk_o_bv64_component_test.cu
    )
    target_include_directories(
      q3x_gdn_prefill_chunk_o_bv64_component_test
      PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/sm87"
    )
    target_link_libraries(
      q3x_gdn_prefill_chunk_o_bv64_component_test
      PRIVATE q3x::kernels CUDA::cudart
    )
    q3x_enable_warnings(
      q3x_gdn_prefill_chunk_o_bv64_component_test
    )
    add_test(NAME gdn_prefill_chunk_o_bv64_component
      COMMAND q3x_gdn_prefill_chunk_o_bv64_component_test
    )
    set_tests_properties(
      gdn_prefill_chunk_o_bv64_component PROPERTIES
      LABELS "cuda;sm87;prefill;gdn;correctness;component"
      SKIP_RETURN_CODE 77
      RESOURCE_LOCK q3x_sm87_gpu
      TIMEOUT 300
    )

    add_executable(
      q3x_gdn_prefill_token_parallel_conv_sm87_test
      tests/gdn_prefill_token_parallel_conv_sm87_test.cu
    )
    target_link_libraries(
      q3x_gdn_prefill_token_parallel_conv_sm87_test
      PRIVATE q3x::kernels CUDA::cudart
    )
    q3x_enable_warnings(
      q3x_gdn_prefill_token_parallel_conv_sm87_test
    )
    add_test(NAME gdn_prefill_token_parallel_conv_sm87
      COMMAND q3x_gdn_prefill_token_parallel_conv_sm87_test
    )
    set_tests_properties(
      gdn_prefill_token_parallel_conv_sm87 PROPERTIES
      LABELS "cuda;sm87;prefill;gdn;correctness;boundary;cuda_graph"
      SKIP_RETURN_CODE 77
      RESOURCE_LOCK q3x_sm87_gpu
      TIMEOUT 300
    )

    add_executable(
      q3x_reference_gdn_prefill_chunk64_native_engine_e2e_test
      tests/reference_gdn_prefill_chunk64_reference_engine_e2e_test.cpp
    )
    target_compile_definitions(
      q3x_reference_gdn_prefill_chunk64_native_engine_e2e_test
      PRIVATE Q3X_GDN_CHUNK64_NATIVE_TEST=1
    )
    target_include_directories(
      q3x_reference_gdn_prefill_chunk64_native_engine_e2e_test
      PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/runtime"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/sm87"
    )
    target_link_libraries(
      q3x_reference_gdn_prefill_chunk64_native_engine_e2e_test
      PRIVATE q3x::engine
    )
    q3x_enable_warnings(
      q3x_reference_gdn_prefill_chunk64_native_engine_e2e_test
    )
    add_test(NAME reference_gdn_prefill_chunk64_native_engine_e2e
      COMMAND q3x_reference_gdn_prefill_chunk64_native_engine_e2e_test
              "${_q3x_e2e_model_dir}"
    )
    set_tests_properties(
      reference_gdn_prefill_chunk64_native_engine_e2e PROPERTIES
      LABELS "e2e;model;sm87;prefill;gdn;architecture;native;characterization"
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

  add_executable(q3x_prefill_execution_plan_test
    tests/prefill_execution_plan_test.cpp
  )
  target_link_libraries(q3x_prefill_execution_plan_test PRIVATE q3x::core)
  q3x_enable_warnings(q3x_prefill_execution_plan_test)
  add_test(NAME prefill_execution_plan
    COMMAND q3x_prefill_execution_plan_test
  )
  set_tests_properties(prefill_execution_plan PROPERTIES
    LABELS "host;prefill;architecture;plan"
  )

  add_executable(q3x_sm87_target_aot_projection_plan_test
    tests/sm87_target_aot_projection_plan_test.cpp
  )
  target_link_libraries(q3x_sm87_target_aot_projection_plan_test
    PRIVATE q3x::core
  )
  q3x_enable_warnings(q3x_sm87_target_aot_projection_plan_test)
  add_test(NAME sm87_target_aot_projection_plan
    COMMAND q3x_sm87_target_aot_projection_plan_test
  )
  set_tests_properties(sm87_target_aot_projection_plan PROPERTIES
    LABELS "host;sm87;prefill;architecture_design;non_executable;projection;plan"
  )

  add_executable(q3x_sm87_target_aot_projection_layout_test
    tests/sm87_target_aot_projection_layout_test.cpp
  )
  target_link_libraries(q3x_sm87_target_aot_projection_layout_test
    PRIVATE q3x::core
  )
  q3x_enable_warnings(q3x_sm87_target_aot_projection_layout_test)
  add_test(NAME sm87_target_aot_projection_layout
    COMMAND q3x_sm87_target_aot_projection_layout_test
  )
  set_tests_properties(sm87_target_aot_projection_layout PROPERTIES
    LABELS "host;sm87;prefill;architecture_design;non_executable;projection;layout"
  )

  add_executable(q3x_sm87_target_aot_attention_plan_test
    tests/sm87_target_aot_attention_plan_test.cpp
  )
  target_link_libraries(q3x_sm87_target_aot_attention_plan_test
    PRIVATE q3x::core
  )
  q3x_enable_warnings(q3x_sm87_target_aot_attention_plan_test)
  add_test(NAME sm87_target_aot_attention_plan
    COMMAND q3x_sm87_target_aot_attention_plan_test
  )
  set_tests_properties(sm87_target_aot_attention_plan PROPERTIES
    LABELS "host;sm87;prefill;architecture_design;non_executable;attention;plan"
  )

  add_executable(q3x_sm87_target_aot_gdn_plan_test
    tests/sm87_target_aot_gdn_plan_test.cpp
  )
  target_link_libraries(q3x_sm87_target_aot_gdn_plan_test
    PRIVATE q3x::core
  )
  q3x_enable_warnings(q3x_sm87_target_aot_gdn_plan_test)
  add_test(NAME sm87_target_aot_gdn_plan
    COMMAND q3x_sm87_target_aot_gdn_plan_test
  )
  set_tests_properties(sm87_target_aot_gdn_plan PROPERTIES
    LABELS "host;sm87;prefill;architecture_design;non_executable;gdn;plan"
  )

  add_executable(q3x_sm87_aot_prefill_system_plan_test
    tests/sm87_aot_prefill_system_plan_test.cpp
  )
  target_link_libraries(q3x_sm87_aot_prefill_system_plan_test
    PRIVATE q3x::core
  )
  q3x_enable_warnings(q3x_sm87_aot_prefill_system_plan_test)
  add_test(NAME sm87_aot_prefill_system_plan
    COMMAND q3x_sm87_aot_prefill_system_plan_test
  )
  set_tests_properties(sm87_aot_prefill_system_plan PROPERTIES
    LABELS "host;sm87;prefill;architecture_design;non_executable;system;plan"
  )

  if(Q3X_BUILD_SM87_AOT_SYSTEM_V1_ARITHMETIC_WITNESS)
    add_executable(q3x_sm87_aot_system_v1_arithmetic_witness_probe
      src/core/sha256.cpp
      src/io/json.cpp
      src/io/safetensors.cpp
      tests/sm87_aot_system_v1_arithmetic_witness_probe.cpp
      tests/support/sm87_aot_system_v1_checkpoint_reader_internal.cpp
      tests/support/sm87_aot_system_v1_arithmetic_witness_internal.cpp
    )
    target_include_directories(
      q3x_sm87_aot_system_v1_arithmetic_witness_probe PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/include"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/support"
    )
    target_compile_features(
      q3x_sm87_aot_system_v1_arithmetic_witness_probe PRIVATE cxx_std_17
    )
    target_compile_definitions(
      q3x_sm87_aot_system_v1_arithmetic_witness_probe PRIVATE
      Q3X_ENABLE_SM87_AOT_SYSTEM_V1_ADMISSION=1
      Q3X_ENABLE_SM87_AOT_SYSTEM_V1_ARITHMETIC_WITNESS=1
    )
    q3x_enable_warnings(q3x_sm87_aot_system_v1_arithmetic_witness_probe)
    add_test(NAME sm87_aot_system_v1_arithmetic_witness_self_test
      COMMAND q3x_sm87_aot_system_v1_arithmetic_witness_probe --self-test
    )
    set_tests_properties(
      sm87_aot_system_v1_arithmetic_witness_self_test PROPERTIES
      LABELS "host;sm87;prefill;target_aot;arithmetic_class;witness;default_off"
    )
  endif()

  if(Q3X_BUILD_SM87_TARGET_AOT_PROJECTION_V1_ADMISSION)
    add_executable(q3x_sm87_target_aot_projection_assets_test
      tests/sm87_target_aot_projection_assets_test.cpp
    )
    target_link_libraries(q3x_sm87_target_aot_projection_assets_test
      PRIVATE q3x::core
    )
    q3x_enable_warnings(q3x_sm87_target_aot_projection_assets_test)
    add_test(NAME sm87_target_aot_projection_assets
      COMMAND q3x_sm87_target_aot_projection_assets_test
    )
    set_tests_properties(sm87_target_aot_projection_assets PROPERTIES
      LABELS "host;sm87;prefill;target_aot;projection;asset;correctness;default_off"
    )

    add_executable(q3x_sm87_target_aot_projection_cuda_compile_test
      tests/sm87_target_aot_projection_cuda_compile_test.cu
    )
    target_link_libraries(q3x_sm87_target_aot_projection_cuda_compile_test
      PRIVATE q3x::kernels CUDA::cudart
    )
    q3x_enable_warnings(q3x_sm87_target_aot_projection_cuda_compile_test)
    add_test(NAME sm87_target_aot_projection_cuda_compile
      COMMAND q3x_sm87_target_aot_projection_cuda_compile_test
    )
    set_tests_properties(sm87_target_aot_projection_cuda_compile PROPERTIES
      LABELS "host;sm87;prefill;target_aot;projection;cuda_compile;default_off"
    )

    if(NOT Q3X_BUILD_SM87_TARGET_AOT_LAYER0_M192_ORACLE_ADMISSION)
      add_test(NAME sm87_target_aot_layer0_m192_oracle_isolation
        COMMAND "${CMAKE_COMMAND}"
          "-DQ3X_KERNEL_ARCHIVE=$<TARGET_FILE:q3x_kernels>"
          "-DQ3X_NM=${CMAKE_NM}"
          -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/VerifySm87TargetAotOracleIsolation.cmake"
      )
      set_tests_properties(
        sm87_target_aot_layer0_m192_oracle_isolation PROPERTIES
        LABELS "host;sm87;prefill;target_aot;projection;oracle_isolation;default_off"
      )
    endif()
  endif()

  if(Q3X_BUILD_SM87_TARGET_AOT_DEVICE_ASSETS_V1_ADMISSION)
    add_executable(q3x_sm87_target_aot_projection_device_assets_host_test
      tests/sm87_target_aot_projection_device_assets_host_test.cpp
    )
    target_link_libraries(
      q3x_sm87_target_aot_projection_device_assets_host_test
      PRIVATE q3x::core
    )
    q3x_enable_warnings(
      q3x_sm87_target_aot_projection_device_assets_host_test
    )
    add_test(NAME sm87_target_aot_projection_device_assets_host
      COMMAND q3x_sm87_target_aot_projection_device_assets_host_test
    )
    set_tests_properties(
      sm87_target_aot_projection_device_assets_host PROPERTIES
      LABELS "host;sm87;prefill;target_aot;projection;device_asset;structure;default_off"
    )

    # Explicitly invoked real-checkpoint preparation probe. It is not a ctest:
    # execution requires an owner-supplied pinned checkpoint, an admitted clean
    # Jetson resource lane, and a repository-local evidence output path.
    find_package(Git REQUIRED)
    set(target_aot_prepare_provenance_header
      "${CMAKE_CURRENT_BINARY_DIR}/generated/tests/sm87_target_aot_prepare_provenance.h"
    )
    add_custom_target(q3x_sm87_target_aot_prepare_provenance
      COMMAND "${CMAKE_COMMAND}"
        "-DQ3X_PROVENANCE_SOURCE_DIR=${CMAKE_SOURCE_DIR}"
        "-DQ3X_PROVENANCE_GIT_EXECUTABLE=${GIT_EXECUTABLE}"
        "-DQ3X_PROVENANCE_TEMPLATE=${CMAKE_SOURCE_DIR}/cmake/sm87_target_aot_prepare_provenance.h.in"
        "-DQ3X_PROVENANCE_OUTPUT=${target_aot_prepare_provenance_header}"
        "-DQ3X_PROVENANCE_CMAKE_VERSION=${CMAKE_VERSION}"
        "-DQ3X_PROVENANCE_GENERATOR=${CMAKE_GENERATOR}"
        "-DQ3X_PROVENANCE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
        "-DQ3X_PROVENANCE_BUILD_TESTING=${BUILD_TESTING}"
        "-DQ3X_PROVENANCE_Q3X_CUDA_ARCHITECTURES=${Q3X_CUDA_ARCHITECTURES}"
        "-DQ3X_PROVENANCE_EFFECTIVE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}"
        "-DQ3X_PROVENANCE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
        "-DQ3X_PROVENANCE_CXX_COMPILER_ID=${CMAKE_CXX_COMPILER_ID}"
        "-DQ3X_PROVENANCE_CXX_COMPILER_VERSION=${CMAKE_CXX_COMPILER_VERSION}"
        "-DQ3X_PROVENANCE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}"
        "-DQ3X_PROVENANCE_CUDA_COMPILER_ID=${CMAKE_CUDA_COMPILER_ID}"
        "-DQ3X_PROVENANCE_CUDA_COMPILER_VERSION=${CMAKE_CUDA_COMPILER_VERSION}"
        "-DQ3X_PROVENANCE_CUDA_TOOLKIT_VERSION=${CUDAToolkit_VERSION}"
        "-DQ3X_PROVENANCE_CXX_RELEASE_FLAGS=${CMAKE_CXX_FLAGS_RELEASE}"
        "-DQ3X_PROVENANCE_CUDA_RELEASE_FLAGS=${CMAKE_CUDA_FLAGS_RELEASE}"
        "-DQ3X_PROVENANCE_CXX_GLOBAL_FLAGS=${CMAKE_CXX_FLAGS}"
        "-DQ3X_PROVENANCE_CUDA_GLOBAL_FLAGS=${CMAKE_CUDA_FLAGS}"
        "-DQ3X_PROVENANCE_AOT_SYSTEM_ADMISSION=${Q3X_BUILD_SM87_AOT_SYSTEM_V1_ADMISSION}"
        "-DQ3X_PROVENANCE_TARGET_PROJECTION_ADMISSION=${Q3X_BUILD_SM87_TARGET_AOT_PROJECTION_V1_ADMISSION}"
        "-DQ3X_PROVENANCE_TARGET_DEVICE_ASSETS_ADMISSION=${Q3X_BUILD_SM87_TARGET_AOT_DEVICE_ASSETS_V1_ADMISSION}"
        "-DQ3X_PROVENANCE_TARGET_LAYER0_M192_ORACLE_ADMISSION=${Q3X_BUILD_SM87_TARGET_AOT_LAYER0_M192_ORACLE_ADMISSION}"
        "-DQ3X_PROVENANCE_FP8_MARLIN_PREFILL_ADMISSION=${Q3X_BUILD_FP8_MARLIN_PREFILL_ADMISSION}"
        "-DQ3X_PROVENANCE_NVFP4_MARLIN_PREFILL_ADMISSION=${Q3X_BUILD_NVFP4_MARLIN_PREFILL_ADMISSION}"
        "-DQ3X_PROVENANCE_P40_PACKED_PROJECTION_ADMISSION=${Q3X_BUILD_P40_PACKED_PROJECTION_ADMISSION}"
        "-DQ3X_PROVENANCE_P40_PACKED_NVFP4_V2_ADMISSION=${Q3X_BUILD_P40_PACKED_NVFP4_V2_ADMISSION}"
        "-DQ3X_PROVENANCE_P40_VLLM_MARLIN_PARITY_ADMISSION=${Q3X_BUILD_P40_VLLM_MARLIN_PARITY_ADMISSION}"
        -P "${CMAKE_SOURCE_DIR}/cmake/GenerateSm87TargetAotPrepareProvenance.cmake"
      BYPRODUCTS "${target_aot_prepare_provenance_header}"
      VERBATIM
    )
    add_executable(q3x_sm87_target_aot_real_checkpoint_prepare_probe
      tests/sm87_target_aot_real_checkpoint_prepare_probe.cpp
    )
    add_dependencies(q3x_sm87_target_aot_real_checkpoint_prepare_probe
      q3x_sm87_target_aot_prepare_provenance
    )
    target_include_directories(
      q3x_sm87_target_aot_real_checkpoint_prepare_probe PRIVATE
      "${CMAKE_CURRENT_BINARY_DIR}/generated/tests"
    )
    if(Q3X_BUILD_SM87_TARGET_AOT_LAYER0_M192_ORACLE_ADMISSION)
      target_include_directories(
        q3x_sm87_target_aot_real_checkpoint_prepare_probe PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/runtime"
      )
    endif()
    target_link_libraries(q3x_sm87_target_aot_real_checkpoint_prepare_probe
      PRIVATE q3x::engine CUDA::cudart
    )
    q3x_enable_warnings(q3x_sm87_target_aot_real_checkpoint_prepare_probe)
    add_test(NAME sm87_target_aot_prepare_probe_contract
      COMMAND "${CMAKE_COMMAND}"
        "-DQ3X_PROBE=$<TARGET_FILE:q3x_sm87_target_aot_real_checkpoint_prepare_probe>"
        "-DQ3X_NM=${CMAKE_NM}"
        "-DQ3X_EXPECT_ORACLE=${Q3X_BUILD_SM87_TARGET_AOT_LAYER0_M192_ORACLE_ADMISSION}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/VerifySm87TargetAotPrepareProbeContract.cmake"
    )
    set_tests_properties(sm87_target_aot_prepare_probe_contract PROPERTIES
      LABELS "host;sm87;prefill;target_aot;checkpoint;oracle;schema;gate"
    )
  endif()

  add_executable(q3x_prefill_operator_bindings_test
    tests/prefill_operator_bindings_test.cpp
  )
  target_link_libraries(q3x_prefill_operator_bindings_test PRIVATE q3x::core)
  q3x_enable_warnings(q3x_prefill_operator_bindings_test)
  add_test(NAME prefill_operator_bindings
    COMMAND q3x_prefill_operator_bindings_test
  )
  set_tests_properties(prefill_operator_bindings PROPERTIES
    LABELS "host;prefill;architecture;bindings"
  )

  add_executable(q3x_prefill_workspace_plan_test
    tests/prefill_workspace_plan_test.cpp
  )
  target_link_libraries(q3x_prefill_workspace_plan_test PRIVATE q3x::core)
  q3x_enable_warnings(q3x_prefill_workspace_plan_test)
  add_test(NAME prefill_workspace_plan
    COMMAND q3x_prefill_workspace_plan_test
  )
  set_tests_properties(prefill_workspace_plan PROPERTIES
    LABELS "host;prefill;architecture;memory;plan"
  )

  add_executable(q3x_gdn_prompt_wide_chunk_graph_plan_test
    tests/gdn_prompt_wide_chunk_graph_plan_test.cpp
  )
  target_link_libraries(q3x_gdn_prompt_wide_chunk_graph_plan_test
    PRIVATE q3x::core
  )
  q3x_enable_warnings(q3x_gdn_prompt_wide_chunk_graph_plan_test)
  add_test(NAME gdn_prompt_wide_chunk_graph_plan
    COMMAND q3x_gdn_prompt_wide_chunk_graph_plan_test
  )
  set_tests_properties(gdn_prompt_wide_chunk_graph_plan PROPERTIES
    LABELS "host;prefill;gdn;architecture_candidate;memory;plan"
  )

  add_executable(q3x_prefill_route_evidence_test
    tests/prefill_route_evidence_test.cpp
  )
  target_link_libraries(q3x_prefill_route_evidence_test PRIVATE q3x::core)
  q3x_enable_warnings(q3x_prefill_route_evidence_test)
  add_test(NAME prefill_route_evidence
    COMMAND q3x_prefill_route_evidence_test
  )
  set_tests_properties(prefill_route_evidence PROPERTIES
    LABELS "host;prefill;route;evidence"
  )

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
  set_tests_properties(generate_help PROPERTIES
    PASS_REGULAR_EXPRESSION "default sm87"
  )
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
  set_tests_properties(benchmark_help PROPERTIES
    PASS_REGULAR_EXPRESSION "default sm87"
  )
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

  if(Q3X_BUILD_P40_WHOLE_CORE_DEVELOPMENT_ROUTE)
    set(q3x_package_expect_eval_server OFF)
  else()
    set(q3x_package_expect_eval_server ON)
  endif()
  add_test(NAME package_consumer
    COMMAND ${CMAKE_COMMAND}
      -DQ3X_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
      -DQ3X_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}
      -DQ3X_GENERATOR=${CMAKE_GENERATOR}
      -DQ3X_CONFIG=$<CONFIG>
      -DQ3X_INSTALL_BINDIR=${CMAKE_INSTALL_BINDIR}
      -DQ3X_EXPECT_EVAL_SERVER=${q3x_package_expect_eval_server}
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
if(NOT Q3X_BUILD_P40_WHOLE_CORE_DEVELOPMENT_ROUTE)
  install(TARGETS qwen3x-eval-server
    EXPORT Qwen3xOrinTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )
endif()
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN "*.h"
  PATTERN "q3x/server/*" EXCLUDE
)
# Remove the header exported by older builds when installing over an existing
# prefix.  The external-reference ABI now lives under tests/support only.
install(CODE
  "file(REMOVE \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/q3x/kernels/sm87_nvfp4_prefill_cublaslt.h\")"
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
