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
option(Q3X_BUILD_GDN_C16_NORM_GATE_ADMISSION
  "Build the test-only exact-C16 GDN norm/gate admission route"
  OFF
)
option(Q3X_BUILD_GDN_CHUNK64_REFERENCE_ADMISSION
  "Build the test-only external-reference C64/WY GDN architecture route"
  OFF
)
option(Q3X_BUILD_GDN_CHUNK64_NATIVE_ADMISSION
  "Build the opt-in native SM87 C64/WY GDN architecture route"
  OFF
)
option(Q3X_BUILD_GDN_PREFILL_PROMPT_SPAN_MACRO_EXPERIMENT
  "Build the isolated SM87 C64 GDN prompt-span macro correctness experiment"
  OFF
)
option(Q3X_BUILD_GDN_PREFILL_PROMPT_SPAN_MACRO_ADMISSION
  "Build the default-off production GDN prompt-span macro route"
  OFF
)
option(Q3X_BUILD_NVFP4_MARLIN_PREFILL_ADMISSION
  "Build the test-only scheduler-wide vLLM-Marlin NVFP4 Prefill route"
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
option(Q3X_BUILD_SM87_A4W4_PRIMITIVE_ADMISSION
  "Build the test-only native SM87 S4xS4 Tensor Core primitive"
  OFF
)
option(Q3X_BUILD_SM87_A4W4_LDMATRIX_OPERAND_PROBE
  "Build the isolated scalar-LDS versus ldmatrix S4 operand-feed probe"
  OFF
)
option(Q3X_BUILD_SM87_A4W4_GATEUP_PAIR_ADMISSION
  "Build the test-only persistent SM87 A4W4 paired Gate+Up route"
  OFF
)
option(Q3X_BUILD_SM87_A4W4_GATEUP_PROJECTION_V3_ADMISSION
  "Build the default-off runtime slice for the warp-specialized Gate+Up v3 cell"
  OFF
)
option(Q3X_BUILD_SM87_A4W4_M128_STAGE_MAJOR_ADMISSION
  "Build the synthetic correctness test for the M128 K128 stage-major Prefill candidate"
  OFF
)
option(Q3X_BUILD_SM87_A4W4_DOWN_K128_STAGE_MAJOR_ADMISSION
  "Build synthetic tests for the Down-specific M128N256 shared-K128 route"
  OFF
)
option(Q3X_BUILD_SM87_A4W4_DOWN_COMPLETE_CELL_V2_ADMISSION
  "Build the test-only runtime admission and tests for the two-CTA Down M128N128K128 complete cell"
  OFF
)
option(Q3X_BUILD_SM87_A4W4_DOWN_COMPLETE_CELL_V3_ADMISSION
  "Build the default-off runtime admission and tests for the Down M32N64 warp-ownership complete cell"
  OFF
)
option(Q3X_BUILD_SM87_A4W4_ATTENTION_SUPERMATRIX_ADMISSION
  "Build the default-off runtime admission and tests for the Attention supermatrix cell"
  OFF
)
option(Q3X_BUILD_SM87_A4W4_ATTENTION_K256_M128N256_ADMISSION
  "Build the default-off K256 M128N256 Attention projection cell"
  OFF
)
option(Q3X_BUILD_SM87_A4W4_ATTENTION_K256_M128N256_A_EXCHANGE_B4_EXPERIMENT
  "Build the default-off 256-thread Attention K256 register-A/B4-ring resource experiment"
  OFF
)
option(Q3X_BUILD_SM87_A4W4_ATTENTION_O_K512_ADMISSION
  "Build the default-off authenticated K512 Attention-O Prefill route"
  OFF
)
option(Q3X_BUILD_SM87_A4W4_MLP_K512_ADMISSION
  "Build the default-off authenticated K512 Gate+Up and Down Prefill route"
  OFF
)
option(Q3X_BUILD_SM87_A4W4_GATEUP_DOWN_K512_EDGE_M128N64_EXPERIMENT
  "Build the default-off producer-owned M128N64 Gate+Up K512 edge candidate"
  OFF
)
option(Q3X_BUILD_SM87_A4W4_GATEUP_DOWN_K512_EDGE_M32N128_PAIR_EXPERIMENT
  "Build the default-off cooperative M32N128 Gate/Up v1 direct-B edge candidate"
  OFF
)
option(Q3X_BUILD_SM87_A4W4_GATEUP_DOWN_K512_EDGE_M32N128_PAIR_SHARED_ACCUM_EXPERIMENT
  "Build the default-off cooperative M32N128 Gate/Up shared-accumulator resource experiment"
  OFF
)
option(Q3X_BUILD_SM87_A4W4_GATEUP_DOWN_K512_EDGE_M64N128_K256_STAGED_EXPERIMENT
  "Build the default-off M64N128 split-projection K256 staged resource experiment"
  OFF
)
option(Q3X_BUILD_SM87_A4W4_GATEUP_DOWN_K512_EDGE_M64N128_K256_ALTERNATING_EXPERIMENT
  "Build the default-off retained-paired M64N128 alternating-K256 resource experiment"
  OFF
)
option(Q3X_BUILD_SM87_A4W4_GATEUP_K512_M128N128_PROJECTION_SERIAL_EXPERIMENT
  "Build the default-off M128N128 full-projection-serial Gate+Up experiment"
  OFF
)
option(Q3X_BUILD_SM87_A4W4_GATEUP_K512_M128N512_FUSED_QUANTIZE_EXPERIMENT
  "Build the default-off M128N512 fused-quantize Gate+Up resource experiment"
  OFF
)
option(Q3X_BUILD_SM87_A4W4_GATEUP_DOWN_K512_EDGE_M128N512_LDMATRIX_ADMISSION
  "Build the default-off M128N512 direct-B LDSM Gate+Up edge admission"
  OFF
)
option(Q3X_BUILD_SM87_A4W4_GATEUP_DOWN_K512_EDGE_M128N512_PAIRED_LDMATRIX_ADMISSION
  "Build the default-off M128N512 paired-B LDSM Gate+Up edge admission"
  OFF
)
option(Q3X_BUILD_SM87_A4W4_DOWN_K512_M128N128_LDMATRIX_PAIRRING_ADMISSION
  "Build the default-off M128N128 LDSM whole-K512 pair-ring Down admission"
  OFF
)
option(Q3X_BUILD_SM87_A4W4_DOWN_K512_M128N128_16WARP_PAIRRING_EXPERIMENT
  "Build the default-off 512-thread M128N128 whole-K512 pair-ring Down experiment"
  OFF
)
option(Q3X_BUILD_SM87_A4W4_DOWN_K512_M16N64_V2_EXPERIMENT
  "Build the default-off 512-thread M16N64 Down K512 candidate"
  OFF
)
option(Q3X_BUILD_SM87_A4W4_MLP_K512_GLOBAL_FLOW_EXPERIMENT
  "Build the rejected shared-memory MLP K512 global-flow experiment"
  OFF
)
option(Q3X_BUILD_SM87_A4W4_MLP_K512_FRAGMENT_NATIVE_EXPERIMENT
  "Build the equal-byte fragment-native MLP K512 kernel experiment"
  OFF
)
option(Q3X_BUILD_SM87_A4W4_GATEUP_K512_FRAGMENT_NATIVE_M128_EXPERIMENT
  "Build the M128-balanced Gate+Up consumer over the fragment-native payload"
  OFF
)
option(Q3X_BUILD_SM87_A4W4_GATEUP_K512_FRAGMENT_NATIVE_M128N64_1CTA_EXPERIMENT
  "Build the direct M128N64 one-CTA Gate+Up consumer over the fragment-native payload"
  OFF
)
option(Q3X_BUILD_SM87_A4W4_GATEUP_K512_FRAGMENT_NATIVE_M64N128_1CTA_EXPERIMENT
  "Build the direct M64N128 16-warp Gate+Up consumer over the fragment-native payload"
  OFF
)
option(Q3X_BUILD_SM87_A4W4_GATEUP_K512_FRAGMENT_NATIVE_M128N64_STAGED_EXPERIMENT
  "Build the staged paired-B M128N64 Gate+Up consumer over the fragment-native payload"
  OFF
)
option(Q3X_BUILD_SM87_A4W4_DOWN_K512_FRAGMENT_NATIVE_M128N256_1CTA_EXPERIMENT
  "Build the direct M128N256 one-CTA Down consumer over the fragment-native payload"
  OFF
)
option(Q3X_BUILD_A4W4_FULL_PREFILL_ADMISSION
  "Build the calibrated full-model SM87 A4W4 Prefill route"
  ON
)
option(Q3X_BUILD_FLASHINFER_PREFILL_ATTENTION_ADMISSION
  "Build the test-only FlashInfer SM87 Prefill-attention route"
  OFF
)
option(Q3X_BUILD_FULL_ATTENTION_LONG_CONTEXT_GROUP_Q64_ADMISSION
  "Build the long-context grouped-Q64 Prefill-attention route"
  ON
)
option(Q3X_BUILD_LONG_PREFILL_LAYER_MAJOR_ADMISSION
  "Build the long-Prefill layer-major executor admission"
  ON
)
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
if(Q3X_BUILD_GDN_PREFILL_PROMPT_SPAN_MACRO_EXPERIMENT AND
   (NOT BUILD_TESTING OR NOT Q3X_BUILD_GDN_CHUNK64_NATIVE_ADMISSION))
  message(FATAL_ERROR
    "The GDN prompt-span macro experiment requires BUILD_TESTING=ON and the authenticated native C64/WY correctness oracle"
  )
endif()
if(Q3X_BUILD_GDN_PREFILL_PROMPT_SPAN_MACRO_ADMISSION AND
   (NOT Q3X_BUILD_GDN_CHUNK64_NATIVE_ADMISSION OR
    NOT Q3X_BUILD_A4W4_FULL_PREFILL_ADMISSION))
  message(FATAL_ERROR
    "The production GDN prompt-span macro requires the native C64 preprocessing route and the A4 whole-prompt runner"
  )
endif()
if(Q3X_BUILD_NVFP4_MARLIN_PREFILL_ADMISSION AND NOT BUILD_TESTING)
  message(FATAL_ERROR
    "Q3X_BUILD_NVFP4_MARLIN_PREFILL_ADMISSION is test-only and requires BUILD_TESTING=ON"
  )
endif()
if(Q3X_BUILD_FP8_MARLIN_PREFILL_ADMISSION AND NOT BUILD_TESTING)
  message(FATAL_ERROR
    "Q3X_BUILD_FP8_MARLIN_PREFILL_ADMISSION is test-only and requires BUILD_TESTING=ON"
  )
endif()
if(Q3X_BUILD_BF16_AB_LARGE_M_PREFILL_ADMISSION AND NOT BUILD_TESTING)
  message(FATAL_ERROR
    "Q3X_BUILD_BF16_AB_LARGE_M_PREFILL_ADMISSION is test-only and requires BUILD_TESTING=ON"
  )
endif()
if(Q3X_BUILD_SM87_A4W4_PRIMITIVE_ADMISSION AND NOT BUILD_TESTING)
  message(FATAL_ERROR
    "Q3X_BUILD_SM87_A4W4_PRIMITIVE_ADMISSION is test-only and requires BUILD_TESTING=ON"
  )
endif()
if(Q3X_BUILD_SM87_A4W4_LDMATRIX_OPERAND_PROBE AND NOT BUILD_TESTING)
  message(FATAL_ERROR
    "Q3X_BUILD_SM87_A4W4_LDMATRIX_OPERAND_PROBE is test-only and requires BUILD_TESTING=ON"
  )
endif()
if(Q3X_BUILD_SM87_A4W4_GATEUP_PAIR_ADMISSION AND NOT BUILD_TESTING)
  message(FATAL_ERROR
    "Q3X_BUILD_SM87_A4W4_GATEUP_PAIR_ADMISSION is test-only and requires BUILD_TESTING=ON"
  )
endif()
if(Q3X_BUILD_SM87_A4W4_GATEUP_PROJECTION_V3_ADMISSION AND
   NOT BUILD_TESTING)
  message(FATAL_ERROR
    "Q3X_BUILD_SM87_A4W4_GATEUP_PROJECTION_V3_ADMISSION is test-only and requires BUILD_TESTING=ON"
  )
endif()
if(Q3X_BUILD_SM87_A4W4_GATEUP_PROJECTION_V3_ADMISSION AND
   NOT Q3X_BUILD_A4W4_FULL_PREFILL_ADMISSION)
  message(FATAL_ERROR
    "The Gate+Up projection v3 runtime slice requires the authenticated full-A4 Prefill route"
  )
endif()
if(Q3X_BUILD_SM87_A4W4_M128_STAGE_MAJOR_ADMISSION AND NOT BUILD_TESTING)
  message(FATAL_ERROR
    "Q3X_BUILD_SM87_A4W4_M128_STAGE_MAJOR_ADMISSION is test-only and requires BUILD_TESTING=ON"
  )
endif()
if(Q3X_BUILD_SM87_A4W4_M128_STAGE_MAJOR_ADMISSION AND
   NOT Q3X_BUILD_A4W4_FULL_PREFILL_ADMISSION)
  message(FATAL_ERROR
    "Q3X_BUILD_SM87_A4W4_M128_STAGE_MAJOR_ADMISSION requires the established K128 comparison kernel"
  )
endif()
if(Q3X_BUILD_SM87_A4W4_DOWN_K128_STAGE_MAJOR_ADMISSION AND
   NOT BUILD_TESTING)
  message(FATAL_ERROR
    "Q3X_BUILD_SM87_A4W4_DOWN_K128_STAGE_MAJOR_ADMISSION is test-only and requires BUILD_TESTING=ON"
  )
endif()
if(Q3X_BUILD_SM87_A4W4_DOWN_K128_STAGE_MAJOR_ADMISSION AND
   NOT Q3X_BUILD_A4W4_FULL_PREFILL_ADMISSION)
  message(FATAL_ERROR
    "The Down K128 stage-major admission compares against the existing full-Prefill K128 implementation"
  )
endif()
if(Q3X_BUILD_SM87_A4W4_DOWN_COMPLETE_CELL_V2_ADMISSION AND
   NOT BUILD_TESTING)
  message(FATAL_ERROR
    "Q3X_BUILD_SM87_A4W4_DOWN_COMPLETE_CELL_V2_ADMISSION is test-only and requires BUILD_TESTING=ON"
  )
endif()
if(Q3X_BUILD_SM87_A4W4_DOWN_COMPLETE_CELL_V2_ADMISSION AND
   NOT Q3X_BUILD_A4W4_FULL_PREFILL_ADMISSION)
  message(FATAL_ERROR
    "The Down complete-cell v2 admission requires the established K128 comparison kernel"
  )
endif()
if(Q3X_BUILD_SM87_A4W4_DOWN_COMPLETE_CELL_V3_ADMISSION AND
   NOT BUILD_TESTING)
  message(FATAL_ERROR
    "Q3X_BUILD_SM87_A4W4_DOWN_COMPLETE_CELL_V3_ADMISSION is test-only and requires BUILD_TESTING=ON"
  )
endif()
if(Q3X_BUILD_SM87_A4W4_DOWN_COMPLETE_CELL_V3_ADMISSION AND
   NOT Q3X_BUILD_A4W4_FULL_PREFILL_ADMISSION)
  message(FATAL_ERROR
    "The Down complete-cell v3 admission requires the established K128 comparison kernel"
  )
endif()
if(Q3X_BUILD_SM87_A4W4_ATTENTION_SUPERMATRIX_ADMISSION AND
   NOT BUILD_TESTING)
  message(FATAL_ERROR
    "Q3X_BUILD_SM87_A4W4_ATTENTION_SUPERMATRIX_ADMISSION requires BUILD_TESTING=ON for its mandatory runtime-admission validation"
  )
endif()
if(Q3X_BUILD_SM87_A4W4_ATTENTION_SUPERMATRIX_ADMISSION AND
   NOT Q3X_BUILD_A4W4_FULL_PREFILL_ADMISSION)
  message(FATAL_ERROR
    "The Attention supermatrix runtime slice requires the authenticated full-A4 Prefill route"
  )
endif()
if(Q3X_BUILD_SM87_A4W4_ATTENTION_K256_M128N256_ADMISSION AND
   NOT BUILD_TESTING)
  message(FATAL_ERROR
    "Q3X_BUILD_SM87_A4W4_ATTENTION_K256_M128N256_ADMISSION requires BUILD_TESTING=ON for mandatory resource and correctness validation"
  )
endif()
if(Q3X_BUILD_SM87_A4W4_ATTENTION_K256_M128N256_ADMISSION AND
   NOT Q3X_BUILD_A4W4_FULL_PREFILL_ADMISSION)
  message(FATAL_ERROR
    "The K256 Attention projection cell requires the authenticated full-A4 Prefill route"
  )
endif()
if(Q3X_BUILD_SM87_A4W4_ATTENTION_K256_M128N256_ADMISSION AND
   NOT Q3X_BUILD_SM87_A4W4_MLP_K512_ADMISSION)
  message(FATAL_ERROR
    "The K256 Attention projection cell requires the complete K512 MLP overlay route because the K256 base has no generic MLP consumer"
  )
endif()
if(Q3X_BUILD_SM87_A4W4_ATTENTION_K256_M128N256_A_EXCHANGE_B4_EXPERIMENT AND
   (NOT BUILD_TESTING OR
    NOT Q3X_BUILD_SM87_A4W4_ATTENTION_K256_M128N256_ADMISSION OR
    NOT Q3X_BUILD_A4W4_FULL_PREFILL_ADMISSION OR
    NOT Q3X_BUILD_SM87_A4W4_MLP_K512_ADMISSION))
  message(FATAL_ERROR
    "The Attention A-exchange/B4 experiment requires BUILD_TESTING plus the authenticated K256 Attention, full-A4, and K512 MLP routes"
  )
endif()
if(Q3X_BUILD_SM87_A4W4_ATTENTION_O_K512_ADMISSION AND
   NOT BUILD_TESTING)
  message(FATAL_ERROR
    "Q3X_BUILD_SM87_A4W4_ATTENTION_O_K512_ADMISSION requires BUILD_TESTING=ON for mandatory bitwise validation"
  )
endif()
if(Q3X_BUILD_SM87_A4W4_ATTENTION_O_K512_ADMISSION AND
   NOT Q3X_BUILD_A4W4_FULL_PREFILL_ADMISSION)
  message(FATAL_ERROR
    "The K512 Attention-O overlay requires the authenticated full-A4 K128 Prefill route"
  )
endif()
if(Q3X_BUILD_SM87_A4W4_MLP_K512_ADMISSION AND NOT BUILD_TESTING)
  message(FATAL_ERROR
    "Q3X_BUILD_SM87_A4W4_MLP_K512_ADMISSION requires BUILD_TESTING=ON for mandatory bitwise validation"
  )
endif()
if(Q3X_BUILD_SM87_A4W4_MLP_K512_ADMISSION AND
   NOT Q3X_BUILD_A4W4_FULL_PREFILL_ADMISSION)
  message(FATAL_ERROR
    "The K512 MLP overlay requires the authenticated full-A4 K128 Prefill route"
  )
endif()
if(Q3X_BUILD_SM87_A4W4_GATEUP_DOWN_K512_EDGE_M128N64_EXPERIMENT AND
   (NOT BUILD_TESTING OR NOT Q3X_BUILD_SM87_A4W4_MLP_K512_ADMISSION))
  message(FATAL_ERROR
    "The M128N64 Gate+Up K512 edge experiment requires BUILD_TESTING=ON and the authenticated MLP K512 route"
  )
endif()
if(Q3X_BUILD_SM87_A4W4_GATEUP_DOWN_K512_EDGE_M32N128_PAIR_EXPERIMENT AND
   (NOT BUILD_TESTING OR NOT Q3X_BUILD_SM87_A4W4_MLP_K512_ADMISSION OR
    NOT Q3X_BUILD_A4W4_FULL_PREFILL_ADMISSION))
  message(FATAL_ERROR
    "The cooperative M32N128 Gate/Up edge experiment requires BUILD_TESTING=ON and the authenticated full-A4 K512 MLP route"
  )
endif()
if(Q3X_BUILD_SM87_A4W4_GATEUP_DOWN_K512_EDGE_M32N128_PAIR_SHARED_ACCUM_EXPERIMENT AND
   (NOT BUILD_TESTING OR NOT Q3X_BUILD_SM87_A4W4_MLP_K512_ADMISSION OR
    NOT Q3X_BUILD_A4W4_FULL_PREFILL_ADMISSION))
  message(FATAL_ERROR
    "The cooperative M32N128 shared-accumulator experiment requires BUILD_TESTING=ON and the authenticated full-A4 K512 MLP route"
  )
endif()
if(Q3X_BUILD_SM87_A4W4_GATEUP_DOWN_K512_EDGE_M64N128_K256_STAGED_EXPERIMENT AND
   (NOT BUILD_TESTING OR NOT Q3X_BUILD_SM87_A4W4_MLP_K512_ADMISSION OR
    NOT Q3X_BUILD_A4W4_FULL_PREFILL_ADMISSION))
  message(FATAL_ERROR
    "The M64N128 split-projection K256 staged experiment requires BUILD_TESTING=ON and the authenticated full-A4 K512 MLP route"
  )
endif()
if(Q3X_BUILD_SM87_A4W4_GATEUP_DOWN_K512_EDGE_M64N128_K256_ALTERNATING_EXPERIMENT AND
   (NOT BUILD_TESTING OR NOT Q3X_BUILD_SM87_A4W4_MLP_K512_ADMISSION OR
    NOT Q3X_BUILD_A4W4_FULL_PREFILL_ADMISSION))
  message(FATAL_ERROR
    "The retained-paired M64N128 alternating-K256 experiment requires BUILD_TESTING=ON and the authenticated full-A4 K512 MLP route"
  )
endif()
if(Q3X_BUILD_SM87_A4W4_GATEUP_K512_M128N128_PROJECTION_SERIAL_EXPERIMENT AND
   (NOT BUILD_TESTING OR NOT Q3X_BUILD_SM87_A4W4_MLP_K512_ADMISSION OR
    NOT Q3X_BUILD_A4W4_FULL_PREFILL_ADMISSION))
  message(FATAL_ERROR
    "The M128N128 projection-serial Gate+Up experiment requires BUILD_TESTING=ON and the authenticated full-A4 K512 MLP route"
  )
endif()
if(Q3X_BUILD_SM87_A4W4_GATEUP_K512_M128N512_FUSED_QUANTIZE_EXPERIMENT AND
   (NOT BUILD_TESTING OR NOT Q3X_BUILD_SM87_A4W4_MLP_K512_ADMISSION OR
    NOT Q3X_BUILD_A4W4_FULL_PREFILL_ADMISSION))
  message(FATAL_ERROR
    "The M128N512 fused-quantize Gate+Up resource experiment requires BUILD_TESTING=ON and the authenticated full-A4 K512 MLP route"
  )
endif()
if(Q3X_BUILD_SM87_A4W4_GATEUP_DOWN_K512_EDGE_M128N512_LDMATRIX_ADMISSION AND
   (NOT BUILD_TESTING OR NOT Q3X_BUILD_SM87_A4W4_MLP_K512_ADMISSION OR
    NOT Q3X_BUILD_A4W4_FULL_PREFILL_ADMISSION))
  message(FATAL_ERROR
    "The M128N512 LDSM Gate+Up edge admission requires BUILD_TESTING=ON and the authenticated full-A4 K512 MLP route"
  )
endif()
if(Q3X_BUILD_SM87_A4W4_GATEUP_DOWN_K512_EDGE_M128N512_PAIRED_LDMATRIX_ADMISSION AND
   (NOT BUILD_TESTING OR NOT Q3X_BUILD_SM87_A4W4_MLP_K512_ADMISSION OR
    NOT Q3X_BUILD_A4W4_FULL_PREFILL_ADMISSION OR
    NOT Q3X_BUILD_SM87_A4W4_MLP_K512_FRAGMENT_NATIVE_EXPERIMENT OR
    NOT Q3X_BUILD_SM87_A4W4_GATEUP_DOWN_K512_EDGE_M128N512_LDMATRIX_ADMISSION))
  message(FATAL_ERROR
    "The paired-B M128N512 LDSM Gate+Up edge admission requires BUILD_TESTING=ON, the authenticated full-A4 K512 MLP route, the composite-layout publication, and the direct-B oracle used by its correctness test"
  )
endif()
if(Q3X_BUILD_SM87_A4W4_DOWN_K512_M128N128_LDMATRIX_PAIRRING_ADMISSION AND
   (NOT BUILD_TESTING OR NOT Q3X_BUILD_SM87_A4W4_MLP_K512_ADMISSION))
  message(FATAL_ERROR
    "The M128N128 LDSM pair-ring Down admission requires BUILD_TESTING=ON and the authenticated MLP K512 route"
  )
endif()
if(Q3X_BUILD_SM87_A4W4_DOWN_K512_M128N128_16WARP_PAIRRING_EXPERIMENT AND
   (NOT BUILD_TESTING OR NOT Q3X_BUILD_SM87_A4W4_MLP_K512_ADMISSION OR
    NOT Q3X_BUILD_A4W4_FULL_PREFILL_ADMISSION OR
    NOT Q3X_BUILD_SM87_A4W4_DOWN_K512_M128N128_LDMATRIX_PAIRRING_ADMISSION))
  message(FATAL_ERROR
    "The 16-warp M128N128 pair-ring Down experiment requires BUILD_TESTING=ON, the authenticated full-A4 K512 MLP route, and the incumbent pair-ring oracle"
  )
endif()
if(Q3X_BUILD_SM87_A4W4_DOWN_K512_M16N64_V2_EXPERIMENT AND
   (NOT BUILD_TESTING OR NOT Q3X_BUILD_SM87_A4W4_MLP_K512_ADMISSION))
  message(FATAL_ERROR
    "The M16N64 Down K512 v2 experiment requires BUILD_TESTING=ON and the authenticated MLP K512 route"
  )
endif()
if(Q3X_BUILD_SM87_A4W4_MLP_K512_GLOBAL_FLOW_EXPERIMENT AND
   (NOT BUILD_TESTING OR NOT Q3X_BUILD_SM87_A4W4_MLP_K512_ADMISSION))
  message(FATAL_ERROR
    "The MLP K512 global-flow experiment requires BUILD_TESTING=ON and the authenticated MLP K512 route"
  )
endif()
if(Q3X_BUILD_SM87_A4W4_MLP_K512_FRAGMENT_NATIVE_EXPERIMENT AND
   (NOT BUILD_TESTING OR NOT Q3X_BUILD_A4W4_FULL_PREFILL_ADMISSION OR
    NOT Q3X_BUILD_SM87_A4W4_MLP_K512_ADMISSION))
  message(FATAL_ERROR
    "The MLP K512 fragment-native experiment requires BUILD_TESTING=ON and the authenticated full-A4 route"
  )
endif()
if(Q3X_BUILD_SM87_A4W4_GATEUP_K512_FRAGMENT_NATIVE_M128_EXPERIMENT AND
   NOT Q3X_BUILD_SM87_A4W4_MLP_K512_FRAGMENT_NATIVE_EXPERIMENT)
  message(FATAL_ERROR
    "The M128 fragment-native Gate+Up experiment requires the authenticated fragment-native MLP route"
  )
endif()
if(Q3X_BUILD_SM87_A4W4_GATEUP_K512_FRAGMENT_NATIVE_M128N64_1CTA_EXPERIMENT AND
   NOT Q3X_BUILD_SM87_A4W4_MLP_K512_FRAGMENT_NATIVE_EXPERIMENT)
  message(FATAL_ERROR
    "The M128N64 one-CTA Gate+Up experiment requires the authenticated fragment-native MLP route"
  )
endif()
if(Q3X_BUILD_SM87_A4W4_GATEUP_K512_FRAGMENT_NATIVE_M64N128_1CTA_EXPERIMENT AND
   NOT Q3X_BUILD_SM87_A4W4_MLP_K512_FRAGMENT_NATIVE_EXPERIMENT)
  message(FATAL_ERROR
    "The M64N128 one-CTA Gate+Up experiment requires the authenticated fragment-native MLP route"
  )
endif()
if(Q3X_BUILD_SM87_A4W4_GATEUP_K512_FRAGMENT_NATIVE_M128N64_STAGED_EXPERIMENT AND
   NOT Q3X_BUILD_SM87_A4W4_MLP_K512_FRAGMENT_NATIVE_EXPERIMENT)
  message(FATAL_ERROR
    "The staged M128N64 Gate+Up experiment requires the authenticated fragment-native MLP route"
  )
endif()
if(Q3X_BUILD_SM87_A4W4_GATEUP_K512_FRAGMENT_NATIVE_M128_EXPERIMENT AND
   Q3X_BUILD_SM87_A4W4_GATEUP_K512_FRAGMENT_NATIVE_M128N64_1CTA_EXPERIMENT)
  message(FATAL_ERROR
    "The M128N32 and M128N64 one-CTA fragment-native Gate+Up experiments are mutually exclusive"
  )
endif()
if(Q3X_BUILD_SM87_A4W4_GATEUP_K512_FRAGMENT_NATIVE_M64N128_1CTA_EXPERIMENT AND
   (Q3X_BUILD_SM87_A4W4_GATEUP_K512_FRAGMENT_NATIVE_M128_EXPERIMENT OR
    Q3X_BUILD_SM87_A4W4_GATEUP_K512_FRAGMENT_NATIVE_M128N64_1CTA_EXPERIMENT))
  message(FATAL_ERROR
    "The M64N128 one-CTA and prior fragment-native Gate+Up experiments are mutually exclusive"
  )
endif()
if(Q3X_BUILD_SM87_A4W4_GATEUP_K512_FRAGMENT_NATIVE_M128N64_STAGED_EXPERIMENT AND
   (Q3X_BUILD_SM87_A4W4_GATEUP_K512_FRAGMENT_NATIVE_M128_EXPERIMENT OR
    Q3X_BUILD_SM87_A4W4_GATEUP_K512_FRAGMENT_NATIVE_M128N64_1CTA_EXPERIMENT OR
    Q3X_BUILD_SM87_A4W4_GATEUP_K512_FRAGMENT_NATIVE_M64N128_1CTA_EXPERIMENT))
  message(FATAL_ERROR
    "The staged M128N64 and prior fragment-native Gate+Up experiments are mutually exclusive"
  )
endif()
if(Q3X_BUILD_SM87_A4W4_DOWN_K512_FRAGMENT_NATIVE_M128N256_1CTA_EXPERIMENT AND
   NOT Q3X_BUILD_SM87_A4W4_MLP_K512_FRAGMENT_NATIVE_EXPERIMENT)
  message(FATAL_ERROR
    "The M128N256 one-CTA Down experiment requires the authenticated fragment-native MLP route"
  )
endif()
if(Q3X_BUILD_FLASHINFER_PREFILL_ATTENTION_ADMISSION AND NOT BUILD_TESTING)
  message(FATAL_ERROR
    "Q3X_BUILD_FLASHINFER_PREFILL_ATTENTION_ADMISSION is test-only and requires BUILD_TESTING=ON"
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
  src/runtime/long_prefill_layer_major.cpp
  src/runtime/prefill_a4_sidecar_converter.cpp
  src/runtime/prefill_attention_o_k512_overlay.cpp
  src/runtime/prefill_mlp_k512_fragment_native_overlay.cpp
  src/runtime/prefill_mlp_k512_overlay.cpp
  src/runtime/prefill_quantized_contract.cpp
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
if(Q3X_BUILD_LONG_PREFILL_LAYER_MAJOR_ADMISSION)
  target_compile_definitions(q3x_core PRIVATE
    Q3X_ENABLE_LONG_PREFILL_LAYER_MAJOR_ADMISSION=1
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
  src/kernels/reference/decode_ops.cu
  src/kernels/reference/full_attention_c512_register_pipeline.cu
  src/kernels/reference/gdn_decode.cu
  src/kernels/reference/gdn_m16_register_state.cu
  src/kernels/reference/gdn_prefill_b8_sequential_sm87.cu
  src/kernels/reference/gdn_prefill_c16_norm_gate_sm87.cu
  src/kernels/reference/layout_ops.cu
  src/kernels/reference/fp8_reference.cu
  src/kernels/reference/gemv_reference.cu
  src/kernels/reference/nvfp4_reference.cu
  src/kernels/reference/projection_dispatch.cpp
  src/kernels/sm87/device_probe.cu
  src/kernels/sm87/decode_gqa_splitkv_sm87.cu
  src/kernels/sm87/fp8_prefill_supermatrix.cu
  src/kernels/sm87/nvfp4_marlin_w4a16.cu
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
if(Q3X_BUILD_A4W4_FULL_PREFILL_ADMISSION)
  target_sources(q3x_kernels PRIVATE
    src/kernels/sm87/a4w4_prefill_primitive.cu
    src/kernels/sm87/a4w4_prefill_gemm.cu
    src/kernels/sm87/a4w4_gateup_paired.cu
    src/kernels/sm87/a4w4_gateup_complete_cell_v2.cu
    src/kernels/sm87/a4w4_prefill_m128_stage_major.cu
    src/kernels/sm87/a4w4_down_k128_stage_major.cu
    src/kernels/sm87/gdn_prefill_exact_span_sm87.cu
    src/kernels/reference/gdn_prefill_whole_span_conv_sm87.cu
  )
  target_compile_definitions(q3x_kernels PRIVATE
    Q3X_ENABLE_A4W4_FULL_PREFILL_ADMISSION=1
  )
endif()
if(Q3X_BUILD_SM87_A4W4_ATTENTION_SUPERMATRIX_ADMISSION)
  target_sources(q3x_kernels PRIVATE
    src/kernels/sm87/a4w4_attention_supermatrix_cell.cu
  )
  target_compile_definitions(q3x_kernels PRIVATE
    Q3X_ENABLE_A4W4_ATTENTION_SUPERMATRIX_ADMISSION=1
  )
endif()
if(Q3X_BUILD_SM87_A4W4_ATTENTION_K256_M128N256_ADMISSION)
  target_sources(q3x_kernels PRIVATE
    src/kernels/sm87/a4w4_attention_k256_m128n256.cu
  )
  target_compile_definitions(q3x_kernels PRIVATE
    Q3X_ENABLE_A4W4_ATTENTION_K256_M128N256_ADMISSION=1
  )
  set_source_files_properties(
    src/kernels/sm87/a4w4_attention_k256_m128n256.cu
    PROPERTIES COMPILE_OPTIONS "--maxrregcount=128;-Xptxas=-warn-spills"
  )
endif()
if(Q3X_BUILD_SM87_A4W4_ATTENTION_K256_M128N256_A_EXCHANGE_B4_EXPERIMENT)
  target_sources(q3x_kernels PRIVATE
    src/kernels/sm87/a4w4_attention_k256_m128n256_a_exchange_b4.cu
  )
  target_compile_definitions(q3x_kernels PRIVATE
    Q3X_ENABLE_A4W4_ATTENTION_K256_M128N256_A_EXCHANGE_B4_EXPERIMENT=1
  )
  set_source_files_properties(
    src/kernels/sm87/a4w4_attention_k256_m128n256_a_exchange_b4.cu
    PROPERTIES COMPILE_OPTIONS "--maxrregcount=255;-Xptxas=-warn-spills"
  )
endif()
if(Q3X_BUILD_SM87_A4W4_ATTENTION_O_K512_ADMISSION)
  target_sources(q3x_kernels PRIVATE
    src/kernels/sm87/a4w4_attention_o_k512_cell.cu
  )
  target_compile_definitions(q3x_kernels PRIVATE
    Q3X_ENABLE_A4W4_ATTENTION_O_K512_ADMISSION=1
  )
endif()
if(Q3X_BUILD_SM87_A4W4_MLP_K512_ADMISSION)
  target_sources(q3x_kernels PRIVATE
    src/kernels/sm87/a4w4_gateup_k512_macrocell.cu
    src/kernels/sm87/a4w4_down_k512_macrocell.cu
    src/kernels/sm87/a4w4_gateup_down_k512_edge.cu
  )
  target_compile_definitions(q3x_kernels PRIVATE
    Q3X_ENABLE_A4W4_MLP_K512_ADMISSION=1
    Q3X_ENABLE_A4W4_GATEUP_DOWN_K512_EDGE_ADMISSION=1
  )
endif()
if(Q3X_BUILD_SM87_A4W4_GATEUP_DOWN_K512_EDGE_M128N64_EXPERIMENT)
  target_sources(q3x_kernels PRIVATE
    src/kernels/sm87/a4w4_gateup_down_k512_edge_m128n64.cu
  )
  target_compile_definitions(q3x_kernels PRIVATE
    Q3X_ENABLE_A4W4_GATEUP_DOWN_K512_EDGE_M128N64_ADMISSION=1
  )
endif()
if(Q3X_BUILD_SM87_A4W4_GATEUP_DOWN_K512_EDGE_M32N128_PAIR_EXPERIMENT)
  target_sources(q3x_kernels PRIVATE
    src/kernels/sm87/a4w4_gateup_down_k512_edge_m32n128_pair.cu
  )
  target_compile_definitions(q3x_kernels PRIVATE
    Q3X_ENABLE_A4W4_GATEUP_DOWN_K512_EDGE_M32N128_PAIR_ADMISSION=1
  )
  set_source_files_properties(
    src/kernels/sm87/a4w4_gateup_down_k512_edge_m32n128_pair.cu
    PROPERTIES COMPILE_OPTIONS "--maxrregcount=128;-Xptxas=-warn-spills"
  )
endif()
if(Q3X_BUILD_SM87_A4W4_GATEUP_DOWN_K512_EDGE_M32N128_PAIR_SHARED_ACCUM_EXPERIMENT)
  target_sources(q3x_kernels PRIVATE
    src/kernels/sm87/a4w4_gateup_down_k512_edge_m32n128_pair_shared_accum.cu
  )
  target_compile_definitions(q3x_kernels PRIVATE
    Q3X_ENABLE_A4W4_GATEUP_DOWN_K512_EDGE_M32N128_PAIR_SHARED_ACCUM_ADMISSION=1
  )
  set_source_files_properties(
    src/kernels/sm87/a4w4_gateup_down_k512_edge_m32n128_pair_shared_accum.cu
    PROPERTIES COMPILE_OPTIONS "--maxrregcount=128;-Xptxas=-warn-spills"
  )
endif()
if(Q3X_BUILD_SM87_A4W4_GATEUP_DOWN_K512_EDGE_M64N128_K256_STAGED_EXPERIMENT)
  target_sources(q3x_kernels PRIVATE
    src/kernels/sm87/a4w4_gateup_down_k512_edge_m64n128_k256_staged.cu
  )
  target_compile_definitions(q3x_kernels PRIVATE
    Q3X_ENABLE_A4W4_GATEUP_DOWN_K512_EDGE_M64N128_K256_STAGED_ADMISSION=1
  )
  set_source_files_properties(
    src/kernels/sm87/a4w4_gateup_down_k512_edge_m64n128_k256_staged.cu
    PROPERTIES COMPILE_OPTIONS "--maxrregcount=128;-Xptxas=-warn-spills"
  )
endif()
if(Q3X_BUILD_SM87_A4W4_GATEUP_DOWN_K512_EDGE_M64N128_K256_ALTERNATING_EXPERIMENT)
  target_sources(q3x_kernels PRIVATE
    src/kernels/sm87/a4w4_gateup_down_k512_edge_m64n128_k256_alternating.cu
  )
  target_compile_definitions(q3x_kernels PRIVATE
    Q3X_ENABLE_A4W4_GATEUP_DOWN_K512_EDGE_M64N128_K256_ALTERNATING_ADMISSION=1
  )
  set_source_files_properties(
    src/kernels/sm87/a4w4_gateup_down_k512_edge_m64n128_k256_alternating.cu
    PROPERTIES COMPILE_OPTIONS "--maxrregcount=128;-Xptxas=-warn-spills"
  )
endif()
if(Q3X_BUILD_SM87_A4W4_GATEUP_K512_M128N128_PROJECTION_SERIAL_EXPERIMENT)
  target_sources(q3x_kernels PRIVATE
    src/kernels/sm87/a4w4_gateup_k512_m128n128_projection_serial.cu
  )
  target_compile_definitions(q3x_kernels PRIVATE
    Q3X_ENABLE_A4W4_GATEUP_K512_M128N128_PROJECTION_SERIAL_ADMISSION=1
  )
  set_source_files_properties(
    src/kernels/sm87/a4w4_gateup_k512_m128n128_projection_serial.cu
    PROPERTIES COMPILE_OPTIONS "--maxrregcount=128;-Xptxas=-v;-Xptxas=-warn-spills"
  )
endif()
if(Q3X_BUILD_SM87_A4W4_GATEUP_K512_M128N512_FUSED_QUANTIZE_EXPERIMENT)
  target_sources(q3x_kernels PRIVATE
    src/kernels/sm87/a4w4_gateup_k512_m128n512_fused_quantize.cu
  )
  target_compile_definitions(q3x_kernels PRIVATE
    Q3X_ENABLE_A4W4_GATEUP_K512_M128N512_FUSED_QUANTIZE_ADMISSION=1
  )
  set_source_files_properties(
    src/kernels/sm87/a4w4_gateup_k512_m128n512_fused_quantize.cu
    PROPERTIES COMPILE_OPTIONS "--maxrregcount=128;-Xptxas=-v;-Xptxas=-warn-spills"
  )
endif()
if(Q3X_BUILD_SM87_A4W4_GATEUP_DOWN_K512_EDGE_M128N512_LDMATRIX_ADMISSION)
  target_sources(q3x_kernels PRIVATE
    src/kernels/sm87/a4w4_gateup_down_k512_edge_m128n512_ldmatrix.cu
  )
  target_compile_definitions(q3x_kernels PRIVATE
    Q3X_ENABLE_A4W4_GATEUP_DOWN_K512_EDGE_M128N512_LDMATRIX_ADMISSION=1
  )
  set_source_files_properties(
    src/kernels/sm87/a4w4_gateup_down_k512_edge_m128n512_ldmatrix.cu
    PROPERTIES COMPILE_OPTIONS "-Xptxas=-v;-Xptxas=-warn-spills"
  )
endif()
if(Q3X_BUILD_SM87_A4W4_GATEUP_DOWN_K512_EDGE_M128N512_PAIRED_LDMATRIX_ADMISSION)
  target_sources(q3x_kernels PRIVATE
    src/kernels/sm87/a4w4_gateup_down_k512_edge_m128n512_paired_ldmatrix.cu
  )
  target_compile_definitions(q3x_kernels PRIVATE
    Q3X_ENABLE_A4W4_GATEUP_DOWN_K512_EDGE_M128N512_PAIRED_LDMATRIX_ADMISSION=1
  )
  set_source_files_properties(
    src/kernels/sm87/a4w4_gateup_down_k512_edge_m128n512_paired_ldmatrix.cu
    PROPERTIES COMPILE_OPTIONS "--maxrregcount=128;-Xptxas=-v;-Xptxas=-warn-spills"
  )
endif()
if(Q3X_BUILD_SM87_A4W4_DOWN_K512_M128N128_LDMATRIX_PAIRRING_ADMISSION)
  target_sources(q3x_kernels PRIVATE
    src/kernels/sm87/a4w4_down_k512_m128n128_ldmatrix_pairring.cu
  )
  target_compile_definitions(q3x_kernels PRIVATE
    Q3X_ENABLE_A4W4_DOWN_K512_M128N128_LDMATRIX_PAIRRING_ADMISSION=1
  )
  set_source_files_properties(
    src/kernels/sm87/a4w4_down_k512_m128n128_ldmatrix_pairring.cu
    PROPERTIES COMPILE_OPTIONS "-Xptxas=-v;-Xptxas=-warn-spills"
  )
endif()
if(Q3X_BUILD_SM87_A4W4_DOWN_K512_M128N128_16WARP_PAIRRING_EXPERIMENT)
  target_sources(q3x_kernels PRIVATE
    src/kernels/sm87/a4w4_down_k512_m128n128_16warp_pairring.cu
  )
  target_compile_definitions(q3x_kernels PRIVATE
    Q3X_ENABLE_A4W4_DOWN_K512_M128N128_16WARP_PAIRRING_ADMISSION=1
  )
  set_source_files_properties(
    src/kernels/sm87/a4w4_down_k512_m128n128_16warp_pairring.cu
    PROPERTIES COMPILE_OPTIONS "--maxrregcount=128;-Xptxas=-v;-Xptxas=-warn-spills"
  )
endif()
if(Q3X_BUILD_SM87_A4W4_DOWN_K512_M16N64_V2_EXPERIMENT)
  target_sources(q3x_kernels PRIVATE
    src/kernels/sm87/a4w4_down_k512_m16n64_v2.cu
  )
  target_compile_definitions(q3x_kernels PRIVATE
    Q3X_ENABLE_A4W4_DOWN_K512_M16N64_V2_ADMISSION=1
  )
endif()
if(Q3X_BUILD_SM87_A4W4_MLP_K512_GLOBAL_FLOW_EXPERIMENT)
  target_sources(q3x_kernels PRIVATE
    src/kernels/sm87/a4w4_gateup_k512_m128n64_global_flow.cu
    src/kernels/sm87/a4w4_down_k512_bstationary.cu
  )
endif()
if(Q3X_BUILD_SM87_A4W4_MLP_K512_FRAGMENT_NATIVE_EXPERIMENT)
  target_sources(q3x_kernels PRIVATE
    src/kernels/sm87/a4w4_gateup_k512_fragment_native.cu
    src/kernels/sm87/a4w4_down_k512_fragment_native.cu
  )
  target_compile_definitions(q3x_kernels PRIVATE
    Q3X_ENABLE_A4W4_MLP_K512_FRAGMENT_NATIVE_ADMISSION=1
  )
  if(Q3X_BUILD_SM87_A4W4_GATEUP_K512_FRAGMENT_NATIVE_M128_EXPERIMENT)
    target_sources(q3x_kernels PRIVATE
      src/kernels/sm87/a4w4_gateup_k512_fragment_native_m128.cu
    )
    target_compile_definitions(q3x_kernels PRIVATE
      Q3X_ENABLE_A4W4_GATEUP_K512_FRAGMENT_NATIVE_M128_ADMISSION=1
    )
  endif()
  if(Q3X_BUILD_SM87_A4W4_GATEUP_K512_FRAGMENT_NATIVE_M128N64_1CTA_EXPERIMENT)
    target_sources(q3x_kernels PRIVATE
      src/kernels/sm87/a4w4_gateup_k512_fragment_native_m128n64_1cta.cu
    )
    target_compile_definitions(q3x_kernels PRIVATE
      Q3X_ENABLE_A4W4_GATEUP_K512_FRAGMENT_NATIVE_M128N64_1CTA_ADMISSION=1
    )
  endif()
  if(Q3X_BUILD_SM87_A4W4_GATEUP_K512_FRAGMENT_NATIVE_M64N128_1CTA_EXPERIMENT)
    target_compile_definitions(q3x_kernels PRIVATE
      Q3X_ENABLE_A4W4_GATEUP_K512_FRAGMENT_NATIVE_M64N128_1CTA_ADMISSION=1
    )
  endif()
  if(Q3X_BUILD_SM87_A4W4_GATEUP_K512_FRAGMENT_NATIVE_M128N64_STAGED_EXPERIMENT)
    target_sources(q3x_kernels PRIVATE
      src/kernels/sm87/a4w4_gateup_k512_fragment_native_m128n64_staged.cu
    )
    target_compile_definitions(q3x_kernels PRIVATE
      Q3X_ENABLE_A4W4_GATEUP_K512_FRAGMENT_NATIVE_M128N64_STAGED_ADMISSION=1
    )
  endif()
  if(Q3X_BUILD_SM87_A4W4_DOWN_K512_FRAGMENT_NATIVE_M128N256_1CTA_EXPERIMENT)
    target_sources(q3x_kernels PRIVATE
      src/kernels/sm87/a4w4_down_k512_fragment_native_m128n256_1cta.cu
    )
    target_compile_definitions(q3x_kernels PRIVATE
      Q3X_ENABLE_A4W4_DOWN_K512_FRAGMENT_NATIVE_M128N256_1CTA_ADMISSION=1
    )
  endif()
endif()
if(Q3X_BUILD_SM87_A4W4_GATEUP_PROJECTION_V3_ADMISSION)
  target_sources(q3x_kernels PRIVATE
    src/kernels/sm87/a4w4_gateup_projection_v3.cu
  )
  target_compile_definitions(q3x_kernels PRIVATE
    Q3X_ENABLE_A4W4_GATEUP_PROJECTION_V3_ADMISSION=1
  )
endif()
if(Q3X_BUILD_SM87_A4W4_DOWN_COMPLETE_CELL_V2_ADMISSION)
  target_sources(q3x_kernels PRIVATE
    src/kernels/sm87/a4w4_down_complete_cell_v2.cu
  )
  target_compile_definitions(q3x_kernels PRIVATE
    Q3X_ENABLE_A4W4_DOWN_COMPLETE_CELL_V2_ADMISSION=1
  )
endif()
if(Q3X_BUILD_SM87_A4W4_DOWN_COMPLETE_CELL_V3_ADMISSION)
  target_sources(q3x_kernels PRIVATE
    src/kernels/sm87/a4w4_down_complete_cell_v3.cu
  )
  target_compile_definitions(q3x_kernels PRIVATE
    Q3X_ENABLE_A4W4_DOWN_COMPLETE_CELL_V3_ADMISSION=1
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
if(Q3X_BUILD_FULL_ATTENTION_LONG_CONTEXT_GROUP_Q64_ADMISSION)
  target_compile_definitions(q3x_kernels PRIVATE
    Q3X_ENABLE_FULL_ATTENTION_LONG_CONTEXT_GROUP_Q64_ADMISSION=1
  )
endif()
if(Q3X_BUILD_GDN_CHUNK64_NATIVE_ADMISSION AND
   NOT Q3X_BUILD_A4W4_FULL_PREFILL_ADMISSION)
  target_sources(q3x_kernels PRIVATE
    src/kernels/reference/gdn_prefill_whole_span_conv_sm87.cu
  )
endif()
if(Q3X_BUILD_GDN_CHUNK64_NATIVE_ADMISSION)
  target_compile_definitions(q3x_kernels PRIVATE
    Q3X_ENABLE_GDN_CHUNK64_NATIVE_ADMISSION=1
  )
endif()
if(Q3X_BUILD_GDN_PREFILL_PROMPT_SPAN_MACRO_ADMISSION)
  target_sources(q3x_kernels PRIVATE
    src/kernels/sm87/gdn_prefill_prompt_span_macro_sm87.cu
  )
  target_compile_definitions(q3x_kernels PRIVATE
    Q3X_ENABLE_GDN_PREFILL_PROMPT_SPAN_MACRO_ADMISSION=1
  )
  set_source_files_properties(
    src/kernels/sm87/gdn_prefill_prompt_span_macro_sm87.cu
    PROPERTIES COMPILE_OPTIONS "-Xptxas=-warn-spills"
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
if(Q3X_BUILD_NVFP4_MARLIN_PREFILL_ADMISSION)
  target_compile_definitions(q3x_engine PRIVATE
    Q3X_ENABLE_NVFP4_MARLIN_PREFILL_ADMISSION=1
  )
endif()
if(Q3X_BUILD_FP8_MARLIN_PREFILL_ADMISSION)
  target_compile_definitions(q3x_engine PRIVATE
    Q3X_ENABLE_FP8_MARLIN_PREFILL_ADMISSION=1
  )
endif()
if(Q3X_BUILD_A4W4_FULL_PREFILL_ADMISSION)
  target_compile_definitions(q3x_engine PRIVATE
    Q3X_ENABLE_A4W4_FULL_PREFILL_ADMISSION=1
  )
endif()
if(Q3X_BUILD_GDN_PREFILL_PROMPT_SPAN_MACRO_ADMISSION)
  target_compile_definitions(q3x_engine PRIVATE
    Q3X_ENABLE_GDN_PREFILL_PROMPT_SPAN_MACRO_ADMISSION=1
  )
endif()
if(Q3X_BUILD_SM87_A4W4_ATTENTION_K256_M128N256_ADMISSION)
  target_compile_definitions(q3x_engine PRIVATE
    Q3X_ENABLE_A4W4_ATTENTION_K256_M128N256_ADMISSION=1
  )
endif()
if(Q3X_BUILD_SM87_A4W4_ATTENTION_K256_M128N256_A_EXCHANGE_B4_EXPERIMENT)
  target_compile_definitions(q3x_engine PRIVATE
    Q3X_ENABLE_A4W4_ATTENTION_K256_M128N256_A_EXCHANGE_B4_EXPERIMENT=1
  )
endif()
if(Q3X_BUILD_SM87_A4W4_ATTENTION_O_K512_ADMISSION)
  target_compile_definitions(q3x_engine PRIVATE
    Q3X_ENABLE_A4W4_ATTENTION_O_K512_ADMISSION=1
  )
endif()
if(Q3X_BUILD_SM87_A4W4_MLP_K512_ADMISSION)
  target_compile_definitions(q3x_engine PRIVATE
    Q3X_ENABLE_A4W4_MLP_K512_ADMISSION=1
    Q3X_ENABLE_A4W4_GATEUP_DOWN_K512_EDGE_ADMISSION=1
  )
endif()
if(Q3X_BUILD_SM87_A4W4_GATEUP_DOWN_K512_EDGE_M128N64_EXPERIMENT)
  target_compile_definitions(q3x_engine PRIVATE
    Q3X_ENABLE_A4W4_GATEUP_DOWN_K512_EDGE_M128N64_ADMISSION=1
  )
endif()
if(Q3X_BUILD_SM87_A4W4_GATEUP_DOWN_K512_EDGE_M64N128_K256_ALTERNATING_EXPERIMENT)
  target_compile_definitions(q3x_engine PRIVATE
    Q3X_ENABLE_A4W4_GATEUP_DOWN_K512_EDGE_M64N128_K256_ALTERNATING_ADMISSION=1
  )
endif()
if(Q3X_BUILD_SM87_A4W4_GATEUP_K512_M128N128_PROJECTION_SERIAL_EXPERIMENT)
  target_compile_definitions(q3x_engine PRIVATE
    Q3X_ENABLE_A4W4_GATEUP_K512_M128N128_PROJECTION_SERIAL_ADMISSION=1
  )
endif()
if(Q3X_BUILD_SM87_A4W4_GATEUP_K512_M128N512_FUSED_QUANTIZE_EXPERIMENT)
  target_compile_definitions(q3x_engine PRIVATE
    Q3X_ENABLE_A4W4_GATEUP_K512_M128N512_FUSED_QUANTIZE_ADMISSION=1
  )
endif()
if(Q3X_BUILD_SM87_A4W4_GATEUP_DOWN_K512_EDGE_M128N512_PAIRED_LDMATRIX_ADMISSION)
  target_compile_definitions(q3x_engine PRIVATE
    Q3X_ENABLE_A4W4_GATEUP_DOWN_K512_EDGE_M128N512_PAIRED_LDMATRIX_ADMISSION=1
  )
endif()
if(Q3X_BUILD_SM87_A4W4_DOWN_K512_M128N128_LDMATRIX_PAIRRING_ADMISSION)
  target_compile_definitions(q3x_engine PRIVATE
    Q3X_ENABLE_A4W4_DOWN_K512_M128N128_LDMATRIX_PAIRRING_ADMISSION=1
  )
endif()
if(Q3X_BUILD_SM87_A4W4_DOWN_K512_M128N128_16WARP_PAIRRING_EXPERIMENT)
  target_compile_definitions(q3x_engine PRIVATE
    Q3X_ENABLE_A4W4_DOWN_K512_M128N128_16WARP_PAIRRING_ADMISSION=1
  )
endif()
if(Q3X_BUILD_SM87_A4W4_DOWN_K512_M16N64_V2_EXPERIMENT)
  target_compile_definitions(q3x_engine PRIVATE
    Q3X_ENABLE_A4W4_DOWN_K512_M16N64_V2_ADMISSION=1
  )
endif()
if(Q3X_BUILD_SM87_A4W4_MLP_K512_FRAGMENT_NATIVE_EXPERIMENT)
  target_compile_definitions(q3x_engine PRIVATE
    Q3X_ENABLE_A4W4_MLP_K512_FRAGMENT_NATIVE_ADMISSION=1
  )
  if(Q3X_BUILD_SM87_A4W4_GATEUP_K512_FRAGMENT_NATIVE_M128_EXPERIMENT)
    target_compile_definitions(q3x_engine PRIVATE
      Q3X_ENABLE_A4W4_GATEUP_K512_FRAGMENT_NATIVE_M128_ADMISSION=1
    )
  endif()
  if(Q3X_BUILD_SM87_A4W4_GATEUP_K512_FRAGMENT_NATIVE_M128N64_1CTA_EXPERIMENT)
    target_compile_definitions(q3x_engine PRIVATE
      Q3X_ENABLE_A4W4_GATEUP_K512_FRAGMENT_NATIVE_M128N64_1CTA_ADMISSION=1
    )
  endif()
  if(Q3X_BUILD_SM87_A4W4_GATEUP_K512_FRAGMENT_NATIVE_M64N128_1CTA_EXPERIMENT)
    target_compile_definitions(q3x_engine PRIVATE
      Q3X_ENABLE_A4W4_GATEUP_K512_FRAGMENT_NATIVE_M64N128_1CTA_ADMISSION=1
    )
  endif()
  if(Q3X_BUILD_SM87_A4W4_GATEUP_K512_FRAGMENT_NATIVE_M128N64_STAGED_EXPERIMENT)
    target_compile_definitions(q3x_engine PRIVATE
      Q3X_ENABLE_A4W4_GATEUP_K512_FRAGMENT_NATIVE_M128N64_STAGED_ADMISSION=1
    )
  endif()
  if(Q3X_BUILD_SM87_A4W4_DOWN_K512_FRAGMENT_NATIVE_M128N256_1CTA_EXPERIMENT)
    target_compile_definitions(q3x_engine PRIVATE
      Q3X_ENABLE_A4W4_DOWN_K512_FRAGMENT_NATIVE_M128N256_1CTA_ADMISSION=1
    )
  endif()
endif()
q3x_enable_warnings(q3x_engine)

# Internal, deliberately narrow OpenAI/EvalScope evaluation adapter. It is
# kept separate from q3x_engine so the core runtime has no HTTP/protocol
# dependency and external evaluation cannot become a hidden production
# fallback path.
add_library(q3x_eval_gateway STATIC
  src/server/openai_protocol.cpp
  src/server/evaluation_server_cli.cpp
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
  # Request-scoped CUDA profiler ranges are diagnostic infrastructure only.
  # Keeping both the implementation define and direct cudart link under
  # BUILD_TESTING prevents it from becoming an implicit production feature.
  target_compile_definitions(q3x_eval_gateway PRIVATE
    Q3X_ENABLE_EVAL_SERVER_CUDA_PROFILER_RANGE=1
  )
  target_link_libraries(q3x_eval_gateway PRIVATE CUDA::cudart)
endif()
q3x_enable_warnings(q3x_eval_gateway)

add_executable(qwen3x-eval-server src/eval_server_main.cpp)
target_link_libraries(qwen3x-eval-server PRIVATE q3x_eval_gateway)
q3x_enable_warnings(qwen3x-eval-server)

add_executable(qwen3x-orin src/main.cpp)
target_link_libraries(qwen3x-orin
  PRIVATE q3x::engine q3x::text q3x::core q3x::kernels
)
q3x_enable_warnings(qwen3x-orin)

add_executable(qwen3x-inspect tools/q3x-inspect/main.cpp)
target_link_libraries(qwen3x-inspect PRIVATE q3x::core)
q3x_enable_warnings(qwen3x-inspect)

add_executable(qwen3x-a4-sidecar tools/q3x-a4-sidecar/main.cpp)
target_link_libraries(qwen3x-a4-sidecar PRIVATE q3x::core)
q3x_enable_warnings(qwen3x-a4-sidecar)

if(BUILD_TESTING)
  find_package(Python3 QUIET COMPONENTS Interpreter)
  if(Python3_Interpreter_FOUND)
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
      NAME long_context_evalscope_harness
      COMMAND "${CMAKE_COMMAND}" -E env "PYTHONDONTWRITEBYTECODE=1"
              "${Python3_EXECUTABLE}"
              "${CMAKE_CURRENT_SOURCE_DIR}/tests/long_context_evalscope_harness_test.py"
    )
    set_tests_properties(long_context_evalscope_harness PROPERTIES
      LABELS "evaluation;host"
    )
  endif()

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

  if(Q3X_BUILD_BF16_AB_LARGE_M_PREFILL_ADMISSION)
    add_executable(q3x_sm87_bf16_ab_prefill_test
      tests/sm87_bf16_ab_prefill_test.cu
    )
    target_link_libraries(q3x_sm87_bf16_ab_prefill_test
      PRIVATE q3x::kernels CUDA::cudart
    )
    q3x_enable_warnings(q3x_sm87_bf16_ab_prefill_test)
    add_test(NAME sm87_bf16_ab_prefill_correctness
      COMMAND q3x_sm87_bf16_ab_prefill_test
    )
    set_tests_properties(sm87_bf16_ab_prefill_correctness PROPERTIES
      LABELS "cuda;sm87;prefill;correctness;synthetic;smoke;admission"
      SKIP_RETURN_CODE 77
      RESOURCE_LOCK q3x_sm87_gpu
      TIMEOUT 120
    )
  endif()

  if(Q3X_BUILD_SM87_A4W4_PRIMITIVE_ADMISSION)
    # Isolated prerequisite for the future full-model A4W4 Prefill plane.
    # Neither this source nor its launch surface is linked into q3x_kernels,
    # the runner, or an installed production target.
    add_library(q3x_test_sm87_a4w4_primitive STATIC
      src/kernels/sm87/a4w4_prefill_primitive.cu
      src/kernels/sm87/a4w4_prefill_gemm.cu
    )
    target_compile_features(q3x_test_sm87_a4w4_primitive
      PUBLIC cxx_std_17 cuda_std_17
    )
    target_include_directories(q3x_test_sm87_a4w4_primitive
      PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
    )
    target_link_libraries(q3x_test_sm87_a4w4_primitive
      PRIVATE CUDA::cudart
    )
    q3x_enable_warnings(q3x_test_sm87_a4w4_primitive)

    add_executable(q3x_sm87_a4w4_primitive_abi_test
      tests/sm87_a4w4_primitive_abi_test.cpp
    )
    target_include_directories(q3x_sm87_a4w4_primitive_abi_test
      PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include"
    )
    target_link_libraries(q3x_sm87_a4w4_primitive_abi_test
      PRIVATE q3x_test_sm87_a4w4_primitive
    )
    q3x_enable_warnings(q3x_sm87_a4w4_primitive_abi_test)
    add_test(NAME sm87_a4w4_primitive_abi
      COMMAND q3x_sm87_a4w4_primitive_abi_test
    )
    set_tests_properties(sm87_a4w4_primitive_abi PROPERTIES
      LABELS "host;sm87;prefill;correctness;synthetic;compile_contract"
    )

    add_executable(q3x_sm87_a4w4_prefill_gemm_test
      tests/sm87_a4w4_prefill_gemm_test.cu
    )
    target_include_directories(q3x_sm87_a4w4_prefill_gemm_test
      PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include"
    )
    target_link_libraries(q3x_sm87_a4w4_prefill_gemm_test
      PRIVATE q3x_test_sm87_a4w4_primitive CUDA::cudart
    )
    q3x_enable_warnings(q3x_sm87_a4w4_prefill_gemm_test)
    add_test(NAME sm87_a4w4_prefill_gemm_correctness
      COMMAND q3x_sm87_a4w4_prefill_gemm_test
    )
    set_tests_properties(sm87_a4w4_prefill_gemm_correctness PROPERTIES
      LABELS "cuda;sm87;prefill;correctness;synthetic;smoke"
      SKIP_RETURN_CODE 77
    )

    # Compile the reusable helper independently to compute_87 PTX and fail the
    # build if nvcc does not retain the exact native S4 Tensor Core opcode.
    set(_q3x_a4w4_ptx_dir
      "${CMAKE_CURRENT_BINARY_DIR}/sm87_a4w4_ptx_sentinel"
    )
    set(_q3x_a4w4_ptx
      "${_q3x_a4w4_ptx_dir}/sm87_a4w4_mma_ptx_sentinel.ptx"
    )
    set(_q3x_a4w4_ptx_stamp
      "${_q3x_a4w4_ptx_dir}/sm87_a4w4_mma_ptx_sentinel.verified"
    )
    add_custom_command(
      OUTPUT "${_q3x_a4w4_ptx}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory
              "${_q3x_a4w4_ptx_dir}"
      COMMAND "${CMAKE_CUDA_COMPILER}"
              --ptx --std=c++17 --gpu-architecture=compute_87
              --include-path "${CMAKE_CURRENT_SOURCE_DIR}/include"
              "${CMAKE_CURRENT_SOURCE_DIR}/tests/sm87_a4w4_mma_ptx_sentinel.cu"
              --output-file "${_q3x_a4w4_ptx}"
      DEPENDS
        tests/sm87_a4w4_mma_ptx_sentinel.cu
        include/q3x/kernels/sm87_a4w4_prefill_primitive.h
      VERBATIM
    )
    add_custom_command(
      OUTPUT "${_q3x_a4w4_ptx_stamp}"
      COMMAND "${CMAKE_COMMAND}"
              "-DQ3X_PTX_FILE=${_q3x_a4w4_ptx}"
              -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/VerifySm87A4W4Ptx.cmake"
      COMMAND "${CMAKE_COMMAND}" -E touch "${_q3x_a4w4_ptx_stamp}"
      DEPENDS
        "${_q3x_a4w4_ptx}"
        cmake/VerifySm87A4W4Ptx.cmake
      VERBATIM
    )
    add_custom_target(q3x_sm87_a4w4_mma_ptx_sentinel ALL
      DEPENDS "${_q3x_a4w4_ptx_stamp}"
    )
  endif()

  if(Q3X_BUILD_SM87_A4W4_LDMATRIX_OPERAND_PROBE)
    # Single-warp P0 only: this executable owns four kernels and the host
    # harness. No source is linked into q3x_kernels or any production runner.
    add_executable(q3x_sm87_a4w4_ldmatrix_operand_probe_test
      src/kernels/sm87/a4w4_ldmatrix_operand_probe.cu
      tests/sm87_a4w4_ldmatrix_operand_probe_correctness_test.cu
    )
    target_include_directories(q3x_sm87_a4w4_ldmatrix_operand_probe_test
      PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include"
    )
    target_link_libraries(q3x_sm87_a4w4_ldmatrix_operand_probe_test
      PRIVATE CUDA::cudart
    )
    target_compile_features(q3x_sm87_a4w4_ldmatrix_operand_probe_test
      PRIVATE cxx_std_17 cuda_std_17
    )
    target_compile_options(q3x_sm87_a4w4_ldmatrix_operand_probe_test PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v;-Xptxas=-warn-spills>
    )
    q3x_enable_warnings(q3x_sm87_a4w4_ldmatrix_operand_probe_test)

    find_program(Q3X_CUOBJDUMP_EXECUTABLE cuobjdump
      HINTS "${CUDAToolkit_BIN_DIR}"
      REQUIRED
    )
    add_test(NAME sm87_a4w4_ldmatrix_operand_probe_binary_contract
      COMMAND "${CMAKE_COMMAND}"
              "-DQ3X_CUOBJDUMP_EXECUTABLE=${Q3X_CUOBJDUMP_EXECUTABLE}"
              "-DQ3X_LDMATRIX_OPERAND_PROBE_BINARY=$<TARGET_FILE:q3x_sm87_a4w4_ldmatrix_operand_probe_test>"
              "-DQ3X_LDMATRIX_OPERAND_PROBE_AUDIT_DIR=${CMAKE_CURRENT_BINARY_DIR}/sm87_a4w4_ldmatrix_operand_probe_audit"
              -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/VerifySm87A4W4LdmatrixOperandProbeBinary.cmake"
    )
    set_tests_properties(
      sm87_a4w4_ldmatrix_operand_probe_binary_contract PROPERTIES
      LABELS "host;sm87;prefill;a4w4;ldmatrix;binary_contract;resources"
    )

    find_program(Q3X_FLOCK_EXECUTABLE flock REQUIRED)
    add_test(NAME sm87_a4w4_ldmatrix_operand_probe_correctness
      COMMAND "${Q3X_FLOCK_EXECUTABLE}" -x /tmp/q3x-gpu-bench.lock
              $<TARGET_FILE:q3x_sm87_a4w4_ldmatrix_operand_probe_test>
    )
    set_tests_properties(
      sm87_a4w4_ldmatrix_operand_probe_correctness PROPERTIES
      LABELS "cuda;sm87;prefill;a4w4;ldmatrix;correctness;synthetic;structural_probe"
      SKIP_RETURN_CODE 77
      RESOURCE_LOCK q3x_sm87_gpu
      TIMEOUT 120
    )
  endif()

  if(Q3X_BUILD_SM87_A4W4_M128_STAGE_MAJOR_ADMISSION)
    # The implementation is linked by the full-A4 production admission above;
    # this option only adds its synthetic ABI/correctness test executable.
    add_executable(q3x_sm87_a4w4_m128_stage_major_test
      tests/sm87_a4w4_prefill_m128_stage_major_test.cu
    )
    target_link_libraries(q3x_sm87_a4w4_m128_stage_major_test
      PRIVATE
        q3x::kernels
        CUDA::cudart
    )
    q3x_enable_warnings(q3x_sm87_a4w4_m128_stage_major_test)
    add_test(NAME sm87_a4w4_m128_stage_major_correctness
      COMMAND q3x_sm87_a4w4_m128_stage_major_test
    )
    set_tests_properties(sm87_a4w4_m128_stage_major_correctness PROPERTIES
      LABELS "cuda;sm87;prefill;correctness;synthetic;structural_candidate"
      SKIP_RETURN_CODE 77
      RESOURCE_LOCK q3x_sm87_gpu
    )
  endif()

  if(Q3X_BUILD_SM87_A4W4_GATEUP_PAIR_ADMISSION)
    # Full paired Gate+Up dataflow admission.  Keep the implementation in a
    # standalone test library so no production runner or selector can link it
    # before the real-model vertical-slice gate is passed.
    add_library(q3x_test_sm87_a4w4_gateup_pair STATIC
      src/kernels/sm87/a4w4_gateup_paired.cu
      src/kernels/sm87/a4w4_gateup_complete_cell_v2.cu
      src/kernels/sm87/a4w4_gateup_projection_v3.cu
    )
    target_compile_features(q3x_test_sm87_a4w4_gateup_pair
      PUBLIC cxx_std_17 cuda_std_17
    )
    target_include_directories(q3x_test_sm87_a4w4_gateup_pair
      PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
    )
    target_link_libraries(q3x_test_sm87_a4w4_gateup_pair
      PRIVATE CUDA::cudart
    )
    q3x_enable_warnings(q3x_test_sm87_a4w4_gateup_pair)

    add_executable(q3x_sm87_a4w4_gateup_pair_contract_test
      tests/sm87_a4w4_gateup_paired_contract_test.cpp
    )
    target_link_libraries(q3x_sm87_a4w4_gateup_pair_contract_test
      PRIVATE q3x_test_sm87_a4w4_gateup_pair CUDA::cudart
    )
    q3x_enable_warnings(q3x_sm87_a4w4_gateup_pair_contract_test)
    add_test(NAME sm87_a4w4_gateup_pair_contract
      COMMAND q3x_sm87_a4w4_gateup_pair_contract_test
    )
    set_tests_properties(sm87_a4w4_gateup_pair_contract PROPERTIES
      LABELS "host;sm87;prefill;correctness;synthetic;compile_contract"
    )

    add_executable(q3x_sm87_a4w4_gateup_pair_correctness_test
      tests/sm87_a4w4_gateup_paired_correctness_test.cu
    )
    target_link_libraries(q3x_sm87_a4w4_gateup_pair_correctness_test
      PRIVATE q3x_test_sm87_a4w4_gateup_pair CUDA::cudart
    )
    q3x_enable_warnings(q3x_sm87_a4w4_gateup_pair_correctness_test)
    add_test(NAME sm87_a4w4_gateup_pair_correctness
      COMMAND q3x_sm87_a4w4_gateup_pair_correctness_test
    )
    set_tests_properties(sm87_a4w4_gateup_pair_correctness PROPERTIES
      LABELS "cuda;sm87;prefill;correctness;synthetic;smoke"
      SKIP_RETURN_CODE 77
      RESOURCE_LOCK q3x_sm87_gpu
    )

    add_executable(q3x_sm87_a4w4_gateup_complete_cell_v2_contract_test
      tests/sm87_a4w4_gateup_complete_cell_v2_contract_test.cpp
    )
    target_link_libraries(
      q3x_sm87_a4w4_gateup_complete_cell_v2_contract_test
      PRIVATE q3x_test_sm87_a4w4_gateup_pair CUDA::cudart
    )
    q3x_enable_warnings(
      q3x_sm87_a4w4_gateup_complete_cell_v2_contract_test
    )
    add_test(NAME sm87_a4w4_gateup_complete_cell_v2_contract
      COMMAND q3x_sm87_a4w4_gateup_complete_cell_v2_contract_test
    )
    set_tests_properties(
      sm87_a4w4_gateup_complete_cell_v2_contract PROPERTIES
      LABELS "host;sm87;prefill;correctness;layout;structural_candidate"
    )

    add_executable(q3x_sm87_a4w4_gateup_complete_cell_v2_correctness_test
      tests/sm87_a4w4_gateup_complete_cell_v2_correctness_test.cu
    )
    target_link_libraries(
      q3x_sm87_a4w4_gateup_complete_cell_v2_correctness_test
      PRIVATE q3x_test_sm87_a4w4_gateup_pair CUDA::cudart
    )
    q3x_enable_warnings(
      q3x_sm87_a4w4_gateup_complete_cell_v2_correctness_test
    )
    add_test(NAME sm87_a4w4_gateup_complete_cell_v2_correctness
      COMMAND q3x_sm87_a4w4_gateup_complete_cell_v2_correctness_test
    )
    set_tests_properties(
      sm87_a4w4_gateup_complete_cell_v2_correctness PROPERTIES
      LABELS "cuda;sm87;prefill;correctness;synthetic;structural_candidate"
      SKIP_RETURN_CODE 77
      RESOURCE_LOCK q3x_sm87_gpu
    )

    add_executable(q3x_sm87_a4w4_gateup_projection_v3_contract_test
      tests/sm87_a4w4_gateup_projection_v3_contract_test.cpp
    )
    target_link_libraries(
      q3x_sm87_a4w4_gateup_projection_v3_contract_test
      PRIVATE q3x_test_sm87_a4w4_gateup_pair CUDA::cudart
    )
    q3x_enable_warnings(
      q3x_sm87_a4w4_gateup_projection_v3_contract_test
    )
    add_test(NAME sm87_a4w4_gateup_projection_v3_contract
      COMMAND q3x_sm87_a4w4_gateup_projection_v3_contract_test
    )
    set_tests_properties(
      sm87_a4w4_gateup_projection_v3_contract PROPERTIES
      LABELS "host;sm87;prefill;correctness;layout;structural_candidate;v3"
    )

    add_executable(q3x_sm87_a4w4_gateup_projection_v3_correctness_test
      tests/sm87_a4w4_gateup_projection_v3_correctness_test.cu
    )
    target_link_libraries(
      q3x_sm87_a4w4_gateup_projection_v3_correctness_test
      PRIVATE q3x_test_sm87_a4w4_gateup_pair CUDA::cudart
    )
    target_compile_options(
      q3x_sm87_a4w4_gateup_projection_v3_correctness_test PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
    )
    q3x_enable_warnings(
      q3x_sm87_a4w4_gateup_projection_v3_correctness_test
    )
    add_test(NAME sm87_a4w4_gateup_projection_v3_correctness
      COMMAND q3x_sm87_a4w4_gateup_projection_v3_correctness_test
    )
    set_tests_properties(
      sm87_a4w4_gateup_projection_v3_correctness PROPERTIES
      LABELS "cuda;sm87;prefill;correctness;synthetic;structural_candidate;v3"
      SKIP_RETURN_CODE 77
      RESOURCE_LOCK q3x_sm87_gpu
      TIMEOUT 180
    )

    # Independently compile the complete kernel to compute_87 PTX and prove
    # that both native S4 MMA and the three-stage cp.async mechanism survive
    # compilation.  Runtime resource admission remains authoritative for
    # occupancy and local-memory/stack use.
    set(_q3x_a4w4_gateup_ptx_dir
      "${CMAKE_CURRENT_BINARY_DIR}/sm87_a4w4_gateup_ptx_sentinel"
    )
    set(_q3x_a4w4_gateup_ptx
      "${_q3x_a4w4_gateup_ptx_dir}/a4w4_gateup_paired.ptx"
    )
    set(_q3x_a4w4_gateup_ptx_stamp
      "${_q3x_a4w4_gateup_ptx_dir}/a4w4_gateup_paired.verified"
    )
    add_custom_command(
      OUTPUT "${_q3x_a4w4_gateup_ptx}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory
              "${_q3x_a4w4_gateup_ptx_dir}"
      COMMAND "${CMAKE_CUDA_COMPILER}"
              --ptx --std=c++17 --gpu-architecture=compute_87
              --include-path "${CMAKE_CURRENT_SOURCE_DIR}/include"
              "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/sm87/a4w4_gateup_paired.cu"
              --output-file "${_q3x_a4w4_gateup_ptx}"
      DEPENDS
        src/kernels/sm87/a4w4_gateup_paired.cu
        include/q3x/kernels/sm87_a4w4_gateup_paired.h
        include/q3x/kernels/sm87_a4w4_prefill_primitive.h
      VERBATIM
    )
    add_custom_command(
      OUTPUT "${_q3x_a4w4_gateup_ptx_stamp}"
      COMMAND "${CMAKE_COMMAND}"
              "-DQ3X_PTX_FILE=${_q3x_a4w4_gateup_ptx}"
              -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/VerifySm87A4W4GateUpPtx.cmake"
      COMMAND "${CMAKE_COMMAND}" -E touch
              "${_q3x_a4w4_gateup_ptx_stamp}"
      DEPENDS
        "${_q3x_a4w4_gateup_ptx}"
        cmake/VerifySm87A4W4GateUpPtx.cmake
      VERBATIM
    )
    add_custom_target(q3x_sm87_a4w4_gateup_ptx_sentinel ALL
      DEPENDS "${_q3x_a4w4_gateup_ptx_stamp}"
    )

    set(_q3x_a4w4_gateup_v3_ptx_dir
      "${CMAKE_CURRENT_BINARY_DIR}/sm87_a4w4_gateup_v3_ptx_sentinel"
    )
    set(_q3x_a4w4_gateup_v3_ptx
      "${_q3x_a4w4_gateup_v3_ptx_dir}/a4w4_gateup_projection_v3.ptx"
    )
    set(_q3x_a4w4_gateup_v3_ptx_stamp
      "${_q3x_a4w4_gateup_v3_ptx_dir}/a4w4_gateup_projection_v3.verified"
    )
    add_custom_command(
      OUTPUT "${_q3x_a4w4_gateup_v3_ptx}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory
              "${_q3x_a4w4_gateup_v3_ptx_dir}"
      COMMAND "${CMAKE_CUDA_COMPILER}"
              --ptx --std=c++17 --gpu-architecture=compute_87
              --include-path "${CMAKE_CURRENT_SOURCE_DIR}/include"
              "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/sm87/a4w4_gateup_projection_v3.cu"
              --output-file "${_q3x_a4w4_gateup_v3_ptx}"
      DEPENDS
        src/kernels/sm87/a4w4_gateup_projection_v3.cu
        include/q3x/kernels/sm87_a4w4_gateup_projection_v3.h
        include/q3x/kernels/sm87_a4w4_prefill_primitive.h
      VERBATIM
    )
    add_custom_command(
      OUTPUT "${_q3x_a4w4_gateup_v3_ptx_stamp}"
      COMMAND "${CMAKE_COMMAND}"
              "-DQ3X_PTX_FILE=${_q3x_a4w4_gateup_v3_ptx}"
              -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/VerifySm87A4W4GateUpPtx.cmake"
      COMMAND "${CMAKE_COMMAND}" -E touch
              "${_q3x_a4w4_gateup_v3_ptx_stamp}"
      DEPENDS
        "${_q3x_a4w4_gateup_v3_ptx}"
        cmake/VerifySm87A4W4GateUpPtx.cmake
      VERBATIM
    )
    add_custom_target(q3x_sm87_a4w4_gateup_v3_ptx_sentinel ALL
      DEPENDS "${_q3x_a4w4_gateup_v3_ptx_stamp}"
    )
  endif()

  if(Q3X_BUILD_SM87_A4W4_DOWN_K128_STAGE_MAJOR_ADMISSION)
    # The implementation has one production linkage through q3x_kernels;
    # this option adds only its focused contract, correctness, and PTX tests.

    add_executable(q3x_sm87_a4w4_down_k128_stage_major_contract_test
      tests/sm87_a4w4_down_k128_stage_major_contract_test.cpp
    )
    target_link_libraries(
      q3x_sm87_a4w4_down_k128_stage_major_contract_test
      PRIVATE q3x::kernels CUDA::cudart
    )
    q3x_enable_warnings(
      q3x_sm87_a4w4_down_k128_stage_major_contract_test
    )
    add_test(NAME sm87_a4w4_down_k128_stage_major_contract
      COMMAND q3x_sm87_a4w4_down_k128_stage_major_contract_test
    )
    set_tests_properties(sm87_a4w4_down_k128_stage_major_contract PROPERTIES
      LABELS "host;sm87;prefill;down;k128;compile_contract"
    )

    add_executable(q3x_sm87_a4w4_down_k128_stage_major_correctness_test
      tests/sm87_a4w4_down_k128_stage_major_correctness_test.cu
    )
    target_link_libraries(
      q3x_sm87_a4w4_down_k128_stage_major_correctness_test
      PRIVATE
        q3x::kernels
        CUDA::cudart
    )
    target_compile_options(
      q3x_sm87_a4w4_down_k128_stage_major_correctness_test PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
    )
    q3x_enable_warnings(
      q3x_sm87_a4w4_down_k128_stage_major_correctness_test
    )
    add_test(NAME sm87_a4w4_down_k128_stage_major_correctness
      COMMAND q3x_sm87_a4w4_down_k128_stage_major_correctness_test
    )
    set_tests_properties(
      sm87_a4w4_down_k128_stage_major_correctness PROPERTIES
      LABELS "cuda;sm87;prefill;down;k128;correctness;synthetic"
      SKIP_RETURN_CODE 77
      RESOURCE_LOCK q3x_sm87_gpu
      TIMEOUT 180
    )

    set(_q3x_down_k128_ptx_dir
      "${CMAKE_CURRENT_BINARY_DIR}/sm87_a4w4_down_k128_ptx_sentinel"
    )
    set(_q3x_down_k128_ptx
      "${_q3x_down_k128_ptx_dir}/a4w4_down_k128_stage_major.ptx"
    )
    set(_q3x_down_k128_ptx_stamp
      "${_q3x_down_k128_ptx_dir}/a4w4_down_k128_stage_major.verified"
    )
    add_custom_command(
      OUTPUT "${_q3x_down_k128_ptx}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory
              "${_q3x_down_k128_ptx_dir}"
      COMMAND "${CMAKE_CUDA_COMPILER}"
              --ptx --std=c++17 --gpu-architecture=compute_87
              --include-path "${CMAKE_CURRENT_SOURCE_DIR}/include"
              "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/sm87/a4w4_down_k128_stage_major.cu"
              --output-file "${_q3x_down_k128_ptx}"
      DEPENDS
        src/kernels/sm87/a4w4_down_k128_stage_major.cu
        include/q3x/kernels/sm87_a4w4_down_k128_stage_major.h
        include/q3x/kernels/sm87_a4w4_prefill_primitive.h
      VERBATIM
    )
    add_custom_command(
      OUTPUT "${_q3x_down_k128_ptx_stamp}"
      COMMAND "${CMAKE_COMMAND}"
              "-DQ3X_PTX_FILE=${_q3x_down_k128_ptx}"
              -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/VerifySm87A4W4GateUpPtx.cmake"
      COMMAND "${CMAKE_COMMAND}" -E touch
              "${_q3x_down_k128_ptx_stamp}"
      DEPENDS
        "${_q3x_down_k128_ptx}"
        cmake/VerifySm87A4W4GateUpPtx.cmake
      VERBATIM
    )
    add_custom_target(q3x_sm87_a4w4_down_k128_ptx_sentinel ALL
      DEPENDS "${_q3x_down_k128_ptx_stamp}"
    )
  endif()

  if(Q3X_BUILD_SM87_A4W4_DOWN_COMPLETE_CELL_V2_ADMISSION)
    # Successor to the rejected M128N256/one-CTA skeleton. This option builds
    # the default-off runtime admission plus its isolated contract/tests.
    add_executable(q3x_sm87_a4w4_down_complete_cell_v2_contract_test
      tests/sm87_a4w4_down_complete_cell_v2_contract_test.cpp
    )
    target_link_libraries(
      q3x_sm87_a4w4_down_complete_cell_v2_contract_test
      PRIVATE q3x::kernels CUDA::cudart
    )
    q3x_enable_warnings(
      q3x_sm87_a4w4_down_complete_cell_v2_contract_test
    )
    add_test(NAME sm87_a4w4_down_complete_cell_v2_contract
      COMMAND q3x_sm87_a4w4_down_complete_cell_v2_contract_test
    )
    set_tests_properties(sm87_a4w4_down_complete_cell_v2_contract PROPERTIES
      LABELS "host;sm87;prefill;down;k128;complete_cell;compile_contract"
    )

    add_executable(q3x_sm87_a4w4_down_complete_cell_v2_correctness_test
      tests/sm87_a4w4_down_complete_cell_v2_correctness_test.cu
    )
    target_link_libraries(
      q3x_sm87_a4w4_down_complete_cell_v2_correctness_test
      PRIVATE q3x::kernels CUDA::cudart
    )
    target_compile_options(
      q3x_sm87_a4w4_down_complete_cell_v2_correctness_test PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
    )
    q3x_enable_warnings(
      q3x_sm87_a4w4_down_complete_cell_v2_correctness_test
    )
    add_test(NAME sm87_a4w4_down_complete_cell_v2_correctness
      COMMAND q3x_sm87_a4w4_down_complete_cell_v2_correctness_test
    )
    set_tests_properties(
      sm87_a4w4_down_complete_cell_v2_correctness PROPERTIES
      LABELS "cuda;sm87;prefill;down;k128;complete_cell;correctness;synthetic"
      SKIP_RETURN_CODE 77
      RESOURCE_LOCK q3x_sm87_gpu
      TIMEOUT 180
    )

    set(_q3x_down_cell_v2_ptx_dir
      "${CMAKE_CURRENT_BINARY_DIR}/sm87_a4w4_down_complete_cell_v2_ptx_sentinel"
    )
    set(_q3x_down_cell_v2_ptx
      "${_q3x_down_cell_v2_ptx_dir}/a4w4_down_complete_cell_v2.ptx"
    )
    set(_q3x_down_cell_v2_ptx_stamp
      "${_q3x_down_cell_v2_ptx_dir}/a4w4_down_complete_cell_v2.verified"
    )
    add_custom_command(
      OUTPUT "${_q3x_down_cell_v2_ptx}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory
              "${_q3x_down_cell_v2_ptx_dir}"
      COMMAND "${CMAKE_CUDA_COMPILER}"
              --ptx --std=c++17 --gpu-architecture=compute_87
              --include-path "${CMAKE_CURRENT_SOURCE_DIR}/include"
              "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/sm87/a4w4_down_complete_cell_v2.cu"
              --output-file "${_q3x_down_cell_v2_ptx}"
      DEPENDS
        src/kernels/sm87/a4w4_down_complete_cell_v2.cu
        include/q3x/kernels/sm87_a4w4_down_complete_cell_v2.h
        include/q3x/kernels/sm87_a4w4_prefill_primitive.h
      VERBATIM
    )
    add_custom_command(
      OUTPUT "${_q3x_down_cell_v2_ptx_stamp}"
      COMMAND "${CMAKE_COMMAND}"
              "-DQ3X_PTX_FILE=${_q3x_down_cell_v2_ptx}"
              -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/VerifySm87A4W4GateUpPtx.cmake"
      COMMAND "${CMAKE_COMMAND}" -E touch
              "${_q3x_down_cell_v2_ptx_stamp}"
      DEPENDS
        "${_q3x_down_cell_v2_ptx}"
        cmake/VerifySm87A4W4GateUpPtx.cmake
      VERBATIM
    )
    add_custom_target(q3x_sm87_a4w4_down_complete_cell_v2_ptx_sentinel ALL
      DEPENDS "${_q3x_down_cell_v2_ptx_stamp}"
    )
  endif()

  if(Q3X_BUILD_SM87_A4W4_DOWN_COMPLETE_CELL_V3_ADMISSION)
    # Successor to complete-cell v2. This option builds the independent,
    # default-off runtime admission plus its isolated contract/tests.
    add_executable(q3x_sm87_a4w4_down_complete_cell_v3_contract_test
      tests/sm87_a4w4_down_complete_cell_v3_contract_test.cpp
    )
    target_link_libraries(
      q3x_sm87_a4w4_down_complete_cell_v3_contract_test
      PRIVATE q3x::kernels CUDA::cudart
    )
    q3x_enable_warnings(
      q3x_sm87_a4w4_down_complete_cell_v3_contract_test
    )
    add_test(NAME sm87_a4w4_down_complete_cell_v3_contract
      COMMAND q3x_sm87_a4w4_down_complete_cell_v3_contract_test
    )
    set_tests_properties(sm87_a4w4_down_complete_cell_v3_contract PROPERTIES
      LABELS "host;sm87;prefill;down;k128;complete_cell;compile_contract"
    )

    add_executable(q3x_sm87_a4w4_down_complete_cell_v3_correctness_test
      tests/sm87_a4w4_down_complete_cell_v3_correctness_test.cu
    )
    target_link_libraries(
      q3x_sm87_a4w4_down_complete_cell_v3_correctness_test
      PRIVATE q3x::kernels CUDA::cudart
    )
    target_compile_options(
      q3x_sm87_a4w4_down_complete_cell_v3_correctness_test PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
    )
    q3x_enable_warnings(
      q3x_sm87_a4w4_down_complete_cell_v3_correctness_test
    )
    add_test(NAME sm87_a4w4_down_complete_cell_v3_correctness
      COMMAND q3x_sm87_a4w4_down_complete_cell_v3_correctness_test
    )
    set_tests_properties(
      sm87_a4w4_down_complete_cell_v3_correctness PROPERTIES
      LABELS "cuda;sm87;prefill;down;k128;complete_cell;correctness;synthetic"
      SKIP_RETURN_CODE 77
      RESOURCE_LOCK q3x_sm87_gpu
      TIMEOUT 180
    )

    set(_q3x_down_cell_v3_ptx_dir
      "${CMAKE_CURRENT_BINARY_DIR}/sm87_a4w4_down_complete_cell_v3_ptx_sentinel"
    )
    set(_q3x_down_cell_v3_ptx
      "${_q3x_down_cell_v3_ptx_dir}/a4w4_down_complete_cell_v3.ptx"
    )
    set(_q3x_down_cell_v3_ptx_stamp
      "${_q3x_down_cell_v3_ptx_dir}/a4w4_down_complete_cell_v3.verified"
    )
    add_custom_command(
      OUTPUT "${_q3x_down_cell_v3_ptx}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory
              "${_q3x_down_cell_v3_ptx_dir}"
      COMMAND "${CMAKE_CUDA_COMPILER}"
              --ptx --std=c++17 --gpu-architecture=compute_87
              --include-path "${CMAKE_CURRENT_SOURCE_DIR}/include"
              "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/sm87/a4w4_down_complete_cell_v3.cu"
              --output-file "${_q3x_down_cell_v3_ptx}"
      DEPENDS
        src/kernels/sm87/a4w4_down_complete_cell_v3.cu
        include/q3x/kernels/sm87_a4w4_down_complete_cell_v3.h
        include/q3x/kernels/sm87_a4w4_prefill_primitive.h
      VERBATIM
    )
    add_custom_command(
      OUTPUT "${_q3x_down_cell_v3_ptx_stamp}"
      COMMAND "${CMAKE_COMMAND}"
              "-DQ3X_PTX_FILE=${_q3x_down_cell_v3_ptx}"
              -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/VerifySm87A4W4GateUpPtx.cmake"
      COMMAND "${CMAKE_COMMAND}" -E touch
              "${_q3x_down_cell_v3_ptx_stamp}"
      DEPENDS
        "${_q3x_down_cell_v3_ptx}"
        cmake/VerifySm87A4W4GateUpPtx.cmake
      VERBATIM
    )
    add_custom_target(q3x_sm87_a4w4_down_complete_cell_v3_ptx_sentinel ALL
      DEPENDS "${_q3x_down_cell_v3_ptx_stamp}"
    )
  endif()

  if(Q3X_BUILD_SM87_A4W4_ATTENTION_SUPERMATRIX_ADMISSION)
    # Keep the structural binary isolated so resource/correctness audits cover
    # only the candidate symbols. The same source is also linked into
    # q3x_kernels behind the independent default-off runtime selector.
    add_library(q3x_test_sm87_a4w4_attention_supermatrix STATIC
      src/kernels/sm87/a4w4_attention_supermatrix_cell.cu
    )
    target_compile_features(q3x_test_sm87_a4w4_attention_supermatrix
      PUBLIC cxx_std_17 cuda_std_17
    )
    target_include_directories(q3x_test_sm87_a4w4_attention_supermatrix
      PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
    )
    target_link_libraries(q3x_test_sm87_a4w4_attention_supermatrix
      PRIVATE CUDA::cudart
    )
    target_compile_options(q3x_test_sm87_a4w4_attention_supermatrix PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-Xptxas=-v>
    )
    q3x_enable_warnings(q3x_test_sm87_a4w4_attention_supermatrix)

    add_executable(q3x_sm87_a4w4_attention_supermatrix_contract_test
      tests/sm87_a4w4_attention_supermatrix_contract_test.cpp
    )
    target_link_libraries(q3x_sm87_a4w4_attention_supermatrix_contract_test
      PRIVATE q3x_test_sm87_a4w4_attention_supermatrix CUDA::cudart
    )
    q3x_enable_warnings(
      q3x_sm87_a4w4_attention_supermatrix_contract_test
    )
    add_test(NAME sm87_a4w4_attention_supermatrix_contract
      COMMAND q3x_sm87_a4w4_attention_supermatrix_contract_test
    )
    set_tests_properties(sm87_a4w4_attention_supermatrix_contract PROPERTIES
      LABELS "host;sm87;prefill;attention;k128;supermatrix;compile_contract"
    )

    find_program(Q3X_CUOBJDUMP_EXECUTABLE cuobjdump
      HINTS "${CUDAToolkit_BIN_DIR}"
      REQUIRED
    )
    add_test(NAME sm87_a4w4_attention_supermatrix_binary_resources
      COMMAND "${CMAKE_COMMAND}"
              "-DQ3X_CUOBJDUMP_EXECUTABLE=${Q3X_CUOBJDUMP_EXECUTABLE}"
              "-DQ3X_ATTENTION_SUPERMATRIX_BINARY=$<TARGET_FILE:q3x_test_sm87_a4w4_attention_supermatrix>"
              -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/VerifySm87A4W4AttentionSupermatrixBinary.cmake"
    )
    set_tests_properties(
      sm87_a4w4_attention_supermatrix_binary_resources PROPERTIES
      LABELS "host;sm87;prefill;attention;k128;supermatrix;binary_resources"
    )

    add_executable(q3x_sm87_a4w4_attention_supermatrix_correctness_test
      tests/sm87_a4w4_attention_supermatrix_correctness_test.cu
    )
    target_link_libraries(
      q3x_sm87_a4w4_attention_supermatrix_correctness_test
      PRIVATE q3x_test_sm87_a4w4_attention_supermatrix CUDA::cudart
    )
    q3x_enable_warnings(
      q3x_sm87_a4w4_attention_supermatrix_correctness_test
    )
    find_program(Q3X_FLOCK_EXECUTABLE flock REQUIRED)
    add_test(NAME sm87_a4w4_attention_supermatrix_correctness
      COMMAND "${Q3X_FLOCK_EXECUTABLE}" -x /tmp/q3x-gpu-bench.lock
              $<TARGET_FILE:q3x_sm87_a4w4_attention_supermatrix_correctness_test>
    )
    set_tests_properties(
      sm87_a4w4_attention_supermatrix_correctness PROPERTIES
      LABELS "cuda;sm87;prefill;attention;k128;supermatrix;correctness;synthetic"
      SKIP_RETURN_CODE 77
      RESOURCE_LOCK q3x_sm87_gpu
      TIMEOUT 180
    )
  endif()

  if(Q3X_BUILD_SM87_A4W4_ATTENTION_K256_M128N256_ADMISSION)
    add_executable(q3x_sm87_a4w4_attention_k256_m128n256_contract_test
      tests/sm87_a4w4_attention_k256_m128n256_contract_test.cpp
    )
    target_link_libraries(
      q3x_sm87_a4w4_attention_k256_m128n256_contract_test
      PRIVATE q3x::kernels CUDA::cudart
    )
    q3x_enable_warnings(
      q3x_sm87_a4w4_attention_k256_m128n256_contract_test
    )
    add_test(NAME sm87_a4w4_attention_k256_m128n256_contract
      COMMAND q3x_sm87_a4w4_attention_k256_m128n256_contract_test
    )
    set_tests_properties(
      sm87_a4w4_attention_k256_m128n256_contract PROPERTIES
      LABELS "host;sm87;prefill;attention;k256;m128n256;compile_contract"
    )

    add_executable(q3x_sm87_a4w4_attention_k256_m128n256_correctness_test
      tests/sm87_a4w4_attention_k256_m128n256_correctness_test.cu
    )
    target_link_libraries(
      q3x_sm87_a4w4_attention_k256_m128n256_correctness_test
      PRIVATE q3x::kernels CUDA::cudart
    )
    q3x_enable_warnings(
      q3x_sm87_a4w4_attention_k256_m128n256_correctness_test
    )
    find_program(Q3X_FLOCK_EXECUTABLE flock REQUIRED)
    add_test(NAME sm87_a4w4_attention_k256_m128n256_correctness
      COMMAND "${Q3X_FLOCK_EXECUTABLE}" -x /tmp/q3x-gpu-bench.lock
              $<TARGET_FILE:q3x_sm87_a4w4_attention_k256_m128n256_correctness_test>
    )
    set_tests_properties(
      sm87_a4w4_attention_k256_m128n256_correctness PROPERTIES
      LABELS "cuda;sm87;prefill;attention;k256;m128n256;correctness;synthetic"
      SKIP_RETURN_CODE 77
      RESOURCE_LOCK q3x_sm87_gpu
      TIMEOUT 240
    )

    if(Q3X_BUILD_SM87_A4W4_ATTENTION_K256_M128N256_A_EXCHANGE_B4_EXPERIMENT)
      add_executable(
        q3x_sm87_a4w4_attention_k256_m128n256_a_exchange_b4_correctness_test
        tests/sm87_a4w4_attention_k256_m128n256_a_exchange_b4_correctness_test.cu
      )
      target_link_libraries(
        q3x_sm87_a4w4_attention_k256_m128n256_a_exchange_b4_correctness_test
        PRIVATE q3x::kernels CUDA::cudart
      )
      q3x_enable_warnings(
        q3x_sm87_a4w4_attention_k256_m128n256_a_exchange_b4_correctness_test
      )
      add_test(
        NAME sm87_a4w4_attention_k256_m128n256_a_exchange_b4_correctness
        COMMAND "${Q3X_FLOCK_EXECUTABLE}" -x /tmp/q3x-gpu-bench.lock
                $<TARGET_FILE:q3x_sm87_a4w4_attention_k256_m128n256_a_exchange_b4_correctness_test>
      )
      set_tests_properties(
        sm87_a4w4_attention_k256_m128n256_a_exchange_b4_correctness
        PROPERTIES
          LABELS "cuda;sm87;prefill;attention;k256;m128n256;structural;correctness;graph"
          SKIP_RETURN_CODE 77
          RESOURCE_LOCK q3x_sm87_gpu
          TIMEOUT 600
      )
    endif()
  endif()

  if(Q3X_BUILD_SM87_A4W4_ATTENTION_O_K512_ADMISSION)
    add_executable(q3x_sm87_a4w4_attention_o_k512_contract_test
      tests/sm87_a4w4_attention_o_k512_contract_test.cpp
    )
    target_link_libraries(q3x_sm87_a4w4_attention_o_k512_contract_test
      PRIVATE q3x::kernels CUDA::cudart
    )
    q3x_enable_warnings(q3x_sm87_a4w4_attention_o_k512_contract_test)
    add_test(NAME sm87_a4w4_attention_o_k512_contract
      COMMAND q3x_sm87_a4w4_attention_o_k512_contract_test
    )
    set_tests_properties(sm87_a4w4_attention_o_k512_contract PROPERTIES
      LABELS "host;sm87;prefill;attention;a4w4;k512;compile_contract"
    )

    add_executable(q3x_prefill_attention_o_k512_overlay_test
      tests/prefill_attention_o_k512_overlay_test.cpp
    )
    target_link_libraries(q3x_prefill_attention_o_k512_overlay_test
      PRIVATE q3x::core
    )
    q3x_enable_warnings(q3x_prefill_attention_o_k512_overlay_test)
    add_test(NAME prefill_attention_o_k512_overlay
      COMMAND q3x_prefill_attention_o_k512_overlay_test
    )
    set_tests_properties(prefill_attention_o_k512_overlay PROPERTIES
      LABELS "host;prefill;attention;a4w4;k512;quantization;sidecar;io"
    )

    add_executable(q3x_sm87_a4w4_attention_o_k512_correctness_test
      tests/sm87_a4w4_attention_o_k512_correctness_test.cu
    )
    target_link_libraries(q3x_sm87_a4w4_attention_o_k512_correctness_test
      PRIVATE q3x::kernels CUDA::cudart
    )
    q3x_enable_warnings(q3x_sm87_a4w4_attention_o_k512_correctness_test)

    add_executable(q3x_sm87_a4_quantize_bf16_k512_correctness_test
      tests/sm87_a4_quantize_bf16_k512_correctness_test.cu
    )
    target_link_libraries(
      q3x_sm87_a4_quantize_bf16_k512_correctness_test
      PRIVATE q3x::kernels CUDA::cudart
    )
    q3x_enable_warnings(
      q3x_sm87_a4_quantize_bf16_k512_correctness_test
    )

    find_program(Q3X_FLOCK_EXECUTABLE flock REQUIRED)
    add_test(NAME sm87_a4w4_attention_o_k512_correctness
      COMMAND "${Q3X_FLOCK_EXECUTABLE}" -x /tmp/q3x-gpu-bench.lock
              $<TARGET_FILE:q3x_sm87_a4w4_attention_o_k512_correctness_test>
    )
    set_tests_properties(sm87_a4w4_attention_o_k512_correctness PROPERTIES
      LABELS "cuda;sm87;prefill;attention;a4w4;k512;correctness;synthetic"
      SKIP_RETURN_CODE 77
      RESOURCE_LOCK q3x_sm87_gpu
      TIMEOUT 300
    )

    add_test(NAME sm87_a4_quantize_bf16_k512_correctness
      COMMAND "${Q3X_FLOCK_EXECUTABLE}" -x /tmp/q3x-gpu-bench.lock
              $<TARGET_FILE:q3x_sm87_a4_quantize_bf16_k512_correctness_test>
    )
    set_tests_properties(
      sm87_a4_quantize_bf16_k512_correctness PROPERTIES
      LABELS "cuda;sm87;prefill;attention;a4w4;k512;quantization;correctness;synthetic"
      SKIP_RETURN_CODE 77
      RESOURCE_LOCK q3x_sm87_gpu
      TIMEOUT 300
    )
  endif()

  if(Q3X_BUILD_SM87_A4W4_MLP_K512_ADMISSION)
    add_executable(q3x_prefill_mlp_k512_overlay_test
      tests/prefill_mlp_k512_overlay_test.cpp
    )
    target_link_libraries(q3x_prefill_mlp_k512_overlay_test
      PRIVATE q3x::core
    )
    q3x_enable_warnings(q3x_prefill_mlp_k512_overlay_test)
    add_test(NAME prefill_mlp_k512_overlay
      COMMAND q3x_prefill_mlp_k512_overlay_test
    )
    set_tests_properties(prefill_mlp_k512_overlay PROPERTIES
      LABELS "host;prefill;mlp;a4w4;k512;quantization;sidecar;io"
    )

    add_executable(q3x_sm87_a4_quantize_bf16_k512_split_correctness_test
      tests/sm87_a4_quantize_bf16_k512_split_correctness_test.cu
    )
    target_link_libraries(
      q3x_sm87_a4_quantize_bf16_k512_split_correctness_test
      PRIVATE q3x::kernels CUDA::cudart
    )
    q3x_enable_warnings(
      q3x_sm87_a4_quantize_bf16_k512_split_correctness_test
    )

    add_executable(q3x_sm87_a4w4_gateup_k512_macrocell_contract_test
      tests/sm87_a4w4_gateup_k512_macrocell_contract_test.cpp
    )
    target_link_libraries(
      q3x_sm87_a4w4_gateup_k512_macrocell_contract_test
      PRIVATE q3x::kernels CUDA::cudart
    )
    q3x_enable_warnings(
      q3x_sm87_a4w4_gateup_k512_macrocell_contract_test
    )
    add_test(NAME sm87_a4w4_gateup_k512_macrocell_contract
      COMMAND q3x_sm87_a4w4_gateup_k512_macrocell_contract_test
    )
    set_tests_properties(
      sm87_a4w4_gateup_k512_macrocell_contract PROPERTIES
      LABELS "host;sm87;prefill;mlp;a4w4;k512;gateup;contract"
    )

    add_executable(q3x_sm87_a4w4_gateup_k512_macrocell_correctness_test
      tests/sm87_a4w4_gateup_k512_macrocell_correctness_test.cu
    )
    target_link_libraries(
      q3x_sm87_a4w4_gateup_k512_macrocell_correctness_test
      PRIVATE q3x::kernels CUDA::cudart
    )
    q3x_enable_warnings(
      q3x_sm87_a4w4_gateup_k512_macrocell_correctness_test
    )

    if(Q3X_BUILD_SM87_A4W4_GATEUP_K512_M128N128_PROJECTION_SERIAL_EXPERIMENT)
      add_executable(
        q3x_sm87_a4w4_gateup_k512_m128n128_projection_serial_contract_test
        tests/sm87_a4w4_gateup_k512_m128n128_projection_serial_contract_test.cpp
      )
      target_link_libraries(
        q3x_sm87_a4w4_gateup_k512_m128n128_projection_serial_contract_test
        PRIVATE q3x::kernels CUDA::cudart
      )
      q3x_enable_warnings(
        q3x_sm87_a4w4_gateup_k512_m128n128_projection_serial_contract_test
      )
      add_test(
        NAME sm87_a4w4_gateup_k512_m128n128_projection_serial_contract
        COMMAND q3x_sm87_a4w4_gateup_k512_m128n128_projection_serial_contract_test
      )
      set_tests_properties(
        sm87_a4w4_gateup_k512_m128n128_projection_serial_contract
        PROPERTIES
        LABELS "host;sm87;prefill;mlp;a4w4;k512;gateup;m128n128;projection-serial;contract"
      )

      add_executable(
        q3x_sm87_a4w4_gateup_k512_m128n128_projection_serial_correctness_test
        tests/sm87_a4w4_gateup_k512_macrocell_correctness_test.cu
      )
      target_compile_definitions(
        q3x_sm87_a4w4_gateup_k512_m128n128_projection_serial_correctness_test
        PRIVATE Q3X_TEST_GATEUP_M128N128_PROJECTION_SERIAL=1
      )
      target_link_libraries(
        q3x_sm87_a4w4_gateup_k512_m128n128_projection_serial_correctness_test
        PRIVATE q3x::kernels CUDA::cudart
      )
      q3x_enable_warnings(
        q3x_sm87_a4w4_gateup_k512_m128n128_projection_serial_correctness_test
      )
    endif()

    if(Q3X_BUILD_SM87_A4W4_GATEUP_K512_M128N512_FUSED_QUANTIZE_EXPERIMENT)
      add_executable(
        q3x_sm87_a4w4_gateup_k512_m128n512_fused_quantize_contract_test
        tests/sm87_a4w4_gateup_k512_m128n512_fused_quantize_contract_test.cpp
      )
      target_link_libraries(
        q3x_sm87_a4w4_gateup_k512_m128n512_fused_quantize_contract_test
        PRIVATE q3x::kernels CUDA::cudart
      )
      q3x_enable_warnings(
        q3x_sm87_a4w4_gateup_k512_m128n512_fused_quantize_contract_test
      )
      add_test(
        NAME sm87_a4w4_gateup_k512_m128n512_fused_quantize_contract
        COMMAND q3x_sm87_a4w4_gateup_k512_m128n512_fused_quantize_contract_test
      )
      set_tests_properties(
        sm87_a4w4_gateup_k512_m128n512_fused_quantize_contract
        PROPERTIES
        LABELS "host;sm87;prefill;mlp;a4w4;k512;gateup;m128n512;fused-quantize;contract;geometry"
      )

      add_executable(
        q3x_sm87_a4w4_gateup_k512_m128n512_fused_quantize_resource_test
        tests/sm87_a4w4_gateup_k512_m128n512_fused_quantize_contract_test.cpp
      )
      target_compile_definitions(
        q3x_sm87_a4w4_gateup_k512_m128n512_fused_quantize_resource_test
        PRIVATE Q3X_TEST_M128N512_FUSED_QUANTIZE_RESOURCES=1
      )
      target_link_libraries(
        q3x_sm87_a4w4_gateup_k512_m128n512_fused_quantize_resource_test
        PRIVATE q3x::kernels CUDA::cudart
      )
      q3x_enable_warnings(
        q3x_sm87_a4w4_gateup_k512_m128n512_fused_quantize_resource_test
      )

      add_executable(
        q3x_sm87_a4w4_gateup_k512_m128n512_fused_quantize_correctness_test
        tests/sm87_a4w4_gateup_k512_m128n512_fused_quantize_correctness_test.cu
      )
      target_link_libraries(
        q3x_sm87_a4w4_gateup_k512_m128n512_fused_quantize_correctness_test
        PRIVATE q3x::kernels CUDA::cudart
      )
      target_compile_options(
        q3x_sm87_a4w4_gateup_k512_m128n512_fused_quantize_correctness_test
        PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
      )
      q3x_enable_warnings(
        q3x_sm87_a4w4_gateup_k512_m128n512_fused_quantize_correctness_test
      )
    endif()

    add_executable(q3x_sm87_a4w4_down_k512_macrocell_contract_test
      tests/sm87_a4w4_down_k512_macrocell_contract_test.cpp
    )
    target_link_libraries(
      q3x_sm87_a4w4_down_k512_macrocell_contract_test
      PRIVATE q3x::kernels CUDA::cudart
    )
    q3x_enable_warnings(
      q3x_sm87_a4w4_down_k512_macrocell_contract_test
    )
    add_test(NAME sm87_a4w4_down_k512_macrocell_contract
      COMMAND q3x_sm87_a4w4_down_k512_macrocell_contract_test
    )
    set_tests_properties(
      sm87_a4w4_down_k512_macrocell_contract PROPERTIES
      LABELS "host;sm87;prefill;mlp;a4w4;k512;down;contract"
    )

    add_executable(q3x_sm87_a4w4_down_k512_macrocell_correctness_test
      tests/sm87_a4w4_down_k512_macrocell_correctness_test.cu
    )
    target_link_libraries(
      q3x_sm87_a4w4_down_k512_macrocell_correctness_test
      PRIVATE q3x::kernels CUDA::cudart
    )
    q3x_enable_warnings(
      q3x_sm87_a4w4_down_k512_macrocell_correctness_test
    )

    add_executable(q3x_sm87_a4w4_gateup_down_k512_edge_contract_test
      tests/sm87_a4w4_gateup_down_k512_edge_contract_test.cpp
    )
    target_link_libraries(
      q3x_sm87_a4w4_gateup_down_k512_edge_contract_test
      PRIVATE q3x::kernels CUDA::cudart
    )
    q3x_enable_warnings(
      q3x_sm87_a4w4_gateup_down_k512_edge_contract_test
    )
    add_test(NAME sm87_a4w4_gateup_down_k512_edge_contract
      COMMAND q3x_sm87_a4w4_gateup_down_k512_edge_contract_test
    )
    set_tests_properties(
      sm87_a4w4_gateup_down_k512_edge_contract PROPERTIES
      LABELS "host;sm87;prefill;mlp;a4w4;k512;gateup;down;edge;contract"
    )

    add_executable(q3x_sm87_a4w4_gateup_down_k512_edge_correctness_test
      tests/sm87_a4w4_gateup_down_k512_edge_correctness_test.cu
    )
    target_link_libraries(
      q3x_sm87_a4w4_gateup_down_k512_edge_correctness_test
      PRIVATE q3x::kernels CUDA::cudart
    )
    q3x_enable_warnings(
      q3x_sm87_a4w4_gateup_down_k512_edge_correctness_test
    )

    if(Q3X_BUILD_SM87_A4W4_GATEUP_DOWN_K512_EDGE_M64N128_K256_ALTERNATING_EXPERIMENT)
      add_executable(
        q3x_sm87_a4w4_gateup_down_k512_edge_m64n128_k256_alternating_contract_test
        tests/sm87_a4w4_gateup_down_k512_edge_m64n128_k256_alternating_contract_test.cpp
      )
      target_link_libraries(
        q3x_sm87_a4w4_gateup_down_k512_edge_m64n128_k256_alternating_contract_test
        PRIVATE q3x::kernels CUDA::cudart
      )
      q3x_enable_warnings(
        q3x_sm87_a4w4_gateup_down_k512_edge_m64n128_k256_alternating_contract_test
      )
      add_test(
        NAME sm87_a4w4_gateup_down_k512_edge_m64n128_k256_alternating_contract
        COMMAND q3x_sm87_a4w4_gateup_down_k512_edge_m64n128_k256_alternating_contract_test
      )
      set_tests_properties(
        sm87_a4w4_gateup_down_k512_edge_m64n128_k256_alternating_contract
        PROPERTIES
        LABELS "host;sm87;prefill;mlp;a4w4;k512;gateup;down;edge;alternating;k256;contract"
      )

      add_executable(
        q3x_sm87_a4w4_gateup_down_k512_edge_m64n128_k256_alternating_correctness_test
        tests/sm87_a4w4_gateup_down_k512_edge_m64n128_k256_alternating_correctness_test.cu
      )
      target_link_libraries(
        q3x_sm87_a4w4_gateup_down_k512_edge_m64n128_k256_alternating_correctness_test
        PRIVATE q3x::kernels CUDA::cudart
      )
      q3x_enable_warnings(
        q3x_sm87_a4w4_gateup_down_k512_edge_m64n128_k256_alternating_correctness_test
      )

      add_executable(
        q3x_sm87_a4w4_gateup_down_k512_edge_m64n128_k256_ldmatrix_pairfeed_correctness_test
        tests/sm87_a4w4_gateup_down_k512_edge_m64n128_k256_alternating_correctness_test.cu
      )
      target_compile_definitions(
        q3x_sm87_a4w4_gateup_down_k512_edge_m64n128_k256_ldmatrix_pairfeed_correctness_test
        PRIVATE Q3X_TEST_GATEUP_LDMATRIX_PAIRFEED=1
      )
      target_link_libraries(
        q3x_sm87_a4w4_gateup_down_k512_edge_m64n128_k256_ldmatrix_pairfeed_correctness_test
        PRIVATE q3x::kernels CUDA::cudart
      )
      q3x_enable_warnings(
        q3x_sm87_a4w4_gateup_down_k512_edge_m64n128_k256_ldmatrix_pairfeed_correctness_test
      )
    endif()

    if(Q3X_BUILD_SM87_A4W4_GATEUP_DOWN_K512_EDGE_M128N512_LDMATRIX_ADMISSION)
      add_executable(
        q3x_sm87_a4w4_gateup_down_k512_edge_m128n512_ldmatrix_contract_test
        tests/sm87_a4w4_gateup_down_k512_edge_m128n512_ldmatrix_contract_test.cpp
      )
      target_link_libraries(
        q3x_sm87_a4w4_gateup_down_k512_edge_m128n512_ldmatrix_contract_test
        PRIVATE q3x::kernels CUDA::cudart
      )
      q3x_enable_warnings(
        q3x_sm87_a4w4_gateup_down_k512_edge_m128n512_ldmatrix_contract_test
      )
      add_test(
        NAME sm87_a4w4_gateup_down_k512_edge_m128n512_ldmatrix_contract
        COMMAND q3x_sm87_a4w4_gateup_down_k512_edge_m128n512_ldmatrix_contract_test
      )
      set_tests_properties(
        sm87_a4w4_gateup_down_k512_edge_m128n512_ldmatrix_contract
        PROPERTIES
        LABELS "host;sm87;prefill;mlp;a4w4;k512;gateup;down;edge;m128n512;ldmatrix;contract"
      )

      add_executable(
        q3x_sm87_a4w4_gateup_down_k512_edge_m128n512_ldmatrix_correctness_test
        tests/sm87_a4w4_gateup_down_k512_edge_m128n512_ldmatrix_correctness_test.cu
      )
      target_link_libraries(
        q3x_sm87_a4w4_gateup_down_k512_edge_m128n512_ldmatrix_correctness_test
        PRIVATE q3x::kernels CUDA::cudart
      )
      q3x_enable_warnings(
        q3x_sm87_a4w4_gateup_down_k512_edge_m128n512_ldmatrix_correctness_test
      )

      find_program(Q3X_CUOBJDUMP_EXECUTABLE cuobjdump
        HINTS "${CUDAToolkit_BIN_DIR}"
        REQUIRED
      )
      add_test(
        NAME sm87_a4w4_gateup_down_k512_edge_m128n512_ldmatrix_binary_contract
        COMMAND "${CMAKE_COMMAND}"
                "-DQ3X_CUOBJDUMP_EXECUTABLE=${Q3X_CUOBJDUMP_EXECUTABLE}"
                "-DQ3X_M128N512_LDMATRIX_BINARY=$<TARGET_FILE:q3x_sm87_a4w4_gateup_down_k512_edge_m128n512_ldmatrix_correctness_test>"
                "-DQ3X_M128N512_LDMATRIX_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/sm87/a4w4_gateup_down_k512_edge_m128n512_ldmatrix.cu"
                "-DQ3X_M128N512_LDMATRIX_AUDIT_DIR=${CMAKE_CURRENT_BINARY_DIR}/sm87_a4w4_gateup_down_k512_edge_m128n512_ldmatrix_audit"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/VerifySm87A4W4GateUpM128N512LdmatrixBinary.cmake"
      )
      set_tests_properties(
        sm87_a4w4_gateup_down_k512_edge_m128n512_ldmatrix_binary_contract
        PROPERTIES
        LABELS "host;sm87;prefill;mlp;a4w4;k512;gateup;edge;m128n512;ldmatrix;binary_contract;resources"
      )
    endif()

    if(Q3X_BUILD_SM87_A4W4_GATEUP_DOWN_K512_EDGE_M128N512_PAIRED_LDMATRIX_ADMISSION)
      add_executable(
        q3x_sm87_a4w4_gateup_down_k512_edge_m128n512_paired_ldmatrix_correctness_test
        tests/sm87_a4w4_gateup_down_k512_edge_m128n512_paired_ldmatrix_correctness_test.cu
      )
      target_link_libraries(
        q3x_sm87_a4w4_gateup_down_k512_edge_m128n512_paired_ldmatrix_correctness_test
        PRIVATE q3x::kernels CUDA::cudart
      )
      q3x_enable_warnings(
        q3x_sm87_a4w4_gateup_down_k512_edge_m128n512_paired_ldmatrix_correctness_test
      )

      find_program(Q3X_CUOBJDUMP_EXECUTABLE cuobjdump
        HINTS "${CUDAToolkit_BIN_DIR}"
        REQUIRED
      )
      add_test(
        NAME sm87_a4w4_gateup_down_k512_edge_m128n512_paired_ldmatrix_binary_contract
        COMMAND "${CMAKE_COMMAND}"
                "-DQ3X_CUOBJDUMP_EXECUTABLE=${Q3X_CUOBJDUMP_EXECUTABLE}"
                "-DQ3X_M128N512_PAIRED_LDMATRIX_BINARY=$<TARGET_FILE:q3x_sm87_a4w4_gateup_down_k512_edge_m128n512_paired_ldmatrix_correctness_test>"
                "-DQ3X_M128N512_PAIRED_LDMATRIX_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/sm87/a4w4_gateup_down_k512_edge_m128n512_paired_ldmatrix.cu"
                "-DQ3X_M128N512_PAIRED_LDMATRIX_AUDIT_DIR=${CMAKE_CURRENT_BINARY_DIR}/sm87_a4w4_gateup_down_k512_edge_m128n512_paired_ldmatrix_audit"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/VerifySm87A4W4GateUpM128N512PairedLdmatrixBinary.cmake"
      )
      set_tests_properties(
        sm87_a4w4_gateup_down_k512_edge_m128n512_paired_ldmatrix_binary_contract
        PROPERTIES
        LABELS "host;sm87;prefill;mlp;a4w4;k512;gateup;edge;m128n512;paired;ldmatrix;binary_contract;resources"
      )
    endif()

    if(Q3X_BUILD_SM87_A4W4_DOWN_K512_M128N128_LDMATRIX_PAIRRING_ADMISSION)
      add_executable(
        q3x_sm87_a4w4_down_k512_m128n128_ldmatrix_pairring_contract_test
        tests/sm87_a4w4_down_k512_m128n128_ldmatrix_pairring_contract_test.cpp
      )
      target_link_libraries(
        q3x_sm87_a4w4_down_k512_m128n128_ldmatrix_pairring_contract_test
        PRIVATE q3x::kernels CUDA::cudart
      )
      q3x_enable_warnings(
        q3x_sm87_a4w4_down_k512_m128n128_ldmatrix_pairring_contract_test
      )
      add_test(
        NAME sm87_a4w4_down_k512_m128n128_ldmatrix_pairring_contract
        COMMAND q3x_sm87_a4w4_down_k512_m128n128_ldmatrix_pairring_contract_test
      )
      set_tests_properties(
        sm87_a4w4_down_k512_m128n128_ldmatrix_pairring_contract
        PROPERTIES
        LABELS "host;sm87;prefill;mlp;a4w4;k512;down;m128n128;ldmatrix;pairring;contract"
      )

      add_executable(
        q3x_sm87_a4w4_down_k512_m128n128_ldmatrix_pairring_correctness_test
        tests/sm87_a4w4_down_k512_m128n128_ldmatrix_pairring_correctness_test.cu
      )
      target_link_libraries(
        q3x_sm87_a4w4_down_k512_m128n128_ldmatrix_pairring_correctness_test
        PRIVATE q3x::kernels CUDA::cudart
      )
      q3x_enable_warnings(
        q3x_sm87_a4w4_down_k512_m128n128_ldmatrix_pairring_correctness_test
      )

      find_program(Q3X_CUOBJDUMP_EXECUTABLE cuobjdump
        HINTS "${CUDAToolkit_BIN_DIR}"
        REQUIRED
      )
      add_test(
        NAME sm87_a4w4_down_k512_m128n128_ldmatrix_pairring_binary_contract
        COMMAND "${CMAKE_COMMAND}"
                "-DQ3X_CUOBJDUMP_EXECUTABLE=${Q3X_CUOBJDUMP_EXECUTABLE}"
                "-DQ3X_DOWN_LDMATRIX_PAIRRING_BINARY=$<TARGET_FILE:q3x_sm87_a4w4_down_k512_m128n128_ldmatrix_pairring_correctness_test>"
                "-DQ3X_DOWN_LDMATRIX_PAIRRING_AUDIT_DIR=${CMAKE_CURRENT_BINARY_DIR}/sm87_a4w4_down_k512_m128n128_ldmatrix_pairring_audit"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/VerifySm87A4W4DownK512M128N128LdmatrixPairringBinary.cmake"
      )
      set_tests_properties(
        sm87_a4w4_down_k512_m128n128_ldmatrix_pairring_binary_contract
        PROPERTIES
        LABELS "host;sm87;prefill;mlp;a4w4;k512;down;m128n128;ldmatrix;pairring;binary_contract;resources"
      )
    endif()

    if(Q3X_BUILD_SM87_A4W4_DOWN_K512_M128N128_16WARP_PAIRRING_EXPERIMENT AND
       Q3X_BUILD_SM87_A4W4_DOWN_K512_M128N128_LDMATRIX_PAIRRING_ADMISSION)
      add_executable(
        q3x_sm87_a4w4_down_k512_m128n128_16warp_pairring_correctness_test
        tests/sm87_a4w4_down_k512_m128n128_16warp_pairring_correctness_test.cu
      )
      target_link_libraries(
        q3x_sm87_a4w4_down_k512_m128n128_16warp_pairring_correctness_test
        PRIVATE q3x::kernels CUDA::cudart
      )
      q3x_enable_warnings(
        q3x_sm87_a4w4_down_k512_m128n128_16warp_pairring_correctness_test
      )
    endif()

    if(Q3X_BUILD_SM87_A4W4_GATEUP_DOWN_K512_EDGE_M128N64_EXPERIMENT)
      add_executable(
        q3x_sm87_a4w4_gateup_down_k512_edge_m128n64_contract_test
        tests/sm87_a4w4_gateup_down_k512_edge_m128n64_contract_test.cpp
      )
      target_link_libraries(
        q3x_sm87_a4w4_gateup_down_k512_edge_m128n64_contract_test
        PRIVATE q3x::kernels CUDA::cudart
      )
      q3x_enable_warnings(
        q3x_sm87_a4w4_gateup_down_k512_edge_m128n64_contract_test
      )
      add_test(NAME sm87_a4w4_gateup_down_k512_edge_m128n64_contract
        COMMAND q3x_sm87_a4w4_gateup_down_k512_edge_m128n64_contract_test
      )
      set_tests_properties(
        sm87_a4w4_gateup_down_k512_edge_m128n64_contract PROPERTIES
        LABELS "host;sm87;prefill;mlp;a4w4;k512;gateup;down;edge;m128n64;contract"
      )

      add_executable(
        q3x_sm87_a4w4_gateup_down_k512_edge_m128n64_correctness_test
        tests/sm87_a4w4_gateup_down_k512_edge_m128n64_correctness_test.cu
      )
      target_link_libraries(
        q3x_sm87_a4w4_gateup_down_k512_edge_m128n64_correctness_test
        PRIVATE q3x::kernels CUDA::cudart
      )
      q3x_enable_warnings(
        q3x_sm87_a4w4_gateup_down_k512_edge_m128n64_correctness_test
      )
    endif()

    if(Q3X_BUILD_SM87_A4W4_DOWN_K512_M16N64_V2_EXPERIMENT)
      add_executable(q3x_sm87_a4w4_down_k512_m16n64_v2_contract_test
        tests/sm87_a4w4_down_k512_m16n64_v2_contract_test.cpp
      )
      target_link_libraries(
        q3x_sm87_a4w4_down_k512_m16n64_v2_contract_test
        PRIVATE q3x::kernels CUDA::cudart
      )
      q3x_enable_warnings(
        q3x_sm87_a4w4_down_k512_m16n64_v2_contract_test
      )
      add_test(NAME sm87_a4w4_down_k512_m16n64_v2_contract
        COMMAND q3x_sm87_a4w4_down_k512_m16n64_v2_contract_test
      )
      set_tests_properties(
        sm87_a4w4_down_k512_m16n64_v2_contract PROPERTIES
        LABELS "host;sm87;prefill;mlp;a4w4;k512;down;v2;contract"
      )

      add_executable(q3x_sm87_a4w4_down_k512_m16n64_v2_correctness_test
        tests/sm87_a4w4_down_k512_m16n64_v2_correctness_test.cu
      )
      target_link_libraries(
        q3x_sm87_a4w4_down_k512_m16n64_v2_correctness_test
        PRIVATE q3x::kernels CUDA::cudart
      )
      q3x_enable_warnings(
        q3x_sm87_a4w4_down_k512_m16n64_v2_correctness_test
      )
    endif()

    if(Q3X_BUILD_SM87_A4W4_MLP_K512_GLOBAL_FLOW_EXPERIMENT)
      add_executable(
      q3x_sm87_a4w4_gateup_k512_m128n64_global_flow_correctness_test
      tests/sm87_a4w4_gateup_k512_m128n64_global_flow_correctness_test.cu
      )
      target_link_libraries(
        q3x_sm87_a4w4_gateup_k512_m128n64_global_flow_correctness_test
        PRIVATE q3x::kernels CUDA::cudart
      )
      q3x_enable_warnings(
        q3x_sm87_a4w4_gateup_k512_m128n64_global_flow_correctness_test
      )

      add_executable(q3x_sm87_a4w4_down_k512_bstationary_contract_test
        tests/sm87_a4w4_down_k512_bstationary_contract_test.cpp
      )
      target_link_libraries(
        q3x_sm87_a4w4_down_k512_bstationary_contract_test
        PRIVATE q3x::kernels CUDA::cudart
      )
      q3x_enable_warnings(
        q3x_sm87_a4w4_down_k512_bstationary_contract_test
      )
      add_test(NAME sm87_a4w4_down_k512_bstationary_contract
        COMMAND q3x_sm87_a4w4_down_k512_bstationary_contract_test
      )
      set_tests_properties(
        sm87_a4w4_down_k512_bstationary_contract PROPERTIES
        LABELS "host;sm87;prefill;mlp;a4w4;k512;down;bstationary;contract"
      )

      add_executable(q3x_sm87_a4w4_down_k512_bstationary_correctness_test
        tests/sm87_a4w4_down_k512_bstationary_correctness_test.cu
      )
      target_link_libraries(
        q3x_sm87_a4w4_down_k512_bstationary_correctness_test
        PRIVATE q3x::kernels CUDA::cudart
      )
      q3x_enable_warnings(
        q3x_sm87_a4w4_down_k512_bstationary_correctness_test
      )
    endif()

    if(Q3X_BUILD_SM87_A4W4_MLP_K512_FRAGMENT_NATIVE_EXPERIMENT)
      add_executable(q3x_prefill_mlp_k512_fragment_native_overlay_test
        tests/prefill_mlp_k512_fragment_native_overlay_test.cpp
      )
      target_link_libraries(
        q3x_prefill_mlp_k512_fragment_native_overlay_test
        PRIVATE q3x::core
      )
      q3x_enable_warnings(
        q3x_prefill_mlp_k512_fragment_native_overlay_test
      )
      add_test(NAME prefill_mlp_k512_fragment_native_overlay
        COMMAND q3x_prefill_mlp_k512_fragment_native_overlay_test
      )
      set_tests_properties(
        prefill_mlp_k512_fragment_native_overlay PROPERTIES
        LABELS "host;prefill;mlp;a4w4;k512;fragment-native;sidecar;io"
      )

      add_executable(q3x_sm87_a4w4_gateup_k512_fragment_native_contract_test
        tests/sm87_a4w4_gateup_k512_fragment_native_contract_test.cpp
      )
      target_link_libraries(
        q3x_sm87_a4w4_gateup_k512_fragment_native_contract_test
        PRIVATE q3x::kernels CUDA::cudart
      )
      q3x_enable_warnings(
        q3x_sm87_a4w4_gateup_k512_fragment_native_contract_test
      )
      add_test(NAME sm87_a4w4_gateup_k512_fragment_native_contract
        COMMAND q3x_sm87_a4w4_gateup_k512_fragment_native_contract_test
      )
      set_tests_properties(
        sm87_a4w4_gateup_k512_fragment_native_contract PROPERTIES
        LABELS "host;sm87;prefill;mlp;a4w4;k512;gateup;fragment-native;contract"
      )

      add_executable(q3x_sm87_a4w4_gateup_k512_fragment_native_correctness_test
        tests/sm87_a4w4_gateup_k512_fragment_native_correctness_test.cu
      )
      target_link_libraries(
        q3x_sm87_a4w4_gateup_k512_fragment_native_correctness_test
        PRIVATE q3x::kernels CUDA::cudart
      )
      q3x_enable_warnings(
        q3x_sm87_a4w4_gateup_k512_fragment_native_correctness_test
      )

      if(Q3X_BUILD_SM87_A4W4_GATEUP_K512_FRAGMENT_NATIVE_M128_EXPERIMENT)
        add_executable(
          q3x_sm87_a4w4_gateup_k512_fragment_native_m128_contract_test
          tests/sm87_a4w4_gateup_k512_fragment_native_m128_contract_test.cpp
        )
        target_link_libraries(
          q3x_sm87_a4w4_gateup_k512_fragment_native_m128_contract_test
          PRIVATE q3x::kernels CUDA::cudart
        )
        q3x_enable_warnings(
          q3x_sm87_a4w4_gateup_k512_fragment_native_m128_contract_test
        )
        add_test(NAME sm87_a4w4_gateup_k512_fragment_native_m128_contract
          COMMAND q3x_sm87_a4w4_gateup_k512_fragment_native_m128_contract_test
        )
        set_tests_properties(
          sm87_a4w4_gateup_k512_fragment_native_m128_contract PROPERTIES
          LABELS "host;sm87;prefill;mlp;a4w4;k512;gateup;fragment-native;m128;contract"
        )

        add_executable(
          q3x_sm87_a4w4_gateup_k512_fragment_native_m128_correctness_test
          tests/sm87_a4w4_gateup_k512_fragment_native_m128_correctness_test.cu
        )
        target_link_libraries(
          q3x_sm87_a4w4_gateup_k512_fragment_native_m128_correctness_test
          PRIVATE q3x::kernels CUDA::cudart
        )
        q3x_enable_warnings(
          q3x_sm87_a4w4_gateup_k512_fragment_native_m128_correctness_test
        )
      endif()

      if(Q3X_BUILD_SM87_A4W4_GATEUP_K512_FRAGMENT_NATIVE_M128N64_1CTA_EXPERIMENT)
        add_executable(
          q3x_sm87_a4w4_gateup_k512_fragment_native_m128n64_1cta_contract_test
          tests/sm87_a4w4_gateup_k512_fragment_native_m128n64_1cta_contract_test.cpp
        )
        target_link_libraries(
          q3x_sm87_a4w4_gateup_k512_fragment_native_m128n64_1cta_contract_test
          PRIVATE q3x::kernels CUDA::cudart
        )
        q3x_enable_warnings(
          q3x_sm87_a4w4_gateup_k512_fragment_native_m128n64_1cta_contract_test
        )
        add_test(NAME sm87_a4w4_gateup_k512_fragment_native_m128n64_1cta_contract
          COMMAND q3x_sm87_a4w4_gateup_k512_fragment_native_m128n64_1cta_contract_test
        )
        set_tests_properties(
          sm87_a4w4_gateup_k512_fragment_native_m128n64_1cta_contract PROPERTIES
          LABELS "host;sm87;prefill;mlp;a4w4;k512;gateup;fragment-native;m128;n64;one-cta;contract"
        )

        add_executable(
          q3x_sm87_a4w4_gateup_k512_fragment_native_m128n64_1cta_correctness_test
          tests/sm87_a4w4_gateup_k512_fragment_native_m128n64_1cta_correctness_test.cu
        )
        target_link_libraries(
          q3x_sm87_a4w4_gateup_k512_fragment_native_m128n64_1cta_correctness_test
          PRIVATE q3x::kernels CUDA::cudart
        )
        q3x_enable_warnings(
          q3x_sm87_a4w4_gateup_k512_fragment_native_m128n64_1cta_correctness_test
        )
      endif()

      if(Q3X_BUILD_SM87_A4W4_GATEUP_K512_FRAGMENT_NATIVE_M64N128_1CTA_EXPERIMENT)
        add_executable(
          q3x_sm87_a4w4_gateup_k512_fragment_native_m64n128_1cta_contract_test
          tests/sm87_a4w4_gateup_k512_fragment_native_m64n128_1cta_contract_test.cpp
        )
        target_link_libraries(
          q3x_sm87_a4w4_gateup_k512_fragment_native_m64n128_1cta_contract_test
          PRIVATE q3x::kernels CUDA::cudart
        )
        q3x_enable_warnings(
          q3x_sm87_a4w4_gateup_k512_fragment_native_m64n128_1cta_contract_test
        )
        add_test(NAME sm87_a4w4_gateup_k512_fragment_native_m64n128_1cta_contract
          COMMAND q3x_sm87_a4w4_gateup_k512_fragment_native_m64n128_1cta_contract_test
        )
        set_tests_properties(
          sm87_a4w4_gateup_k512_fragment_native_m64n128_1cta_contract PROPERTIES
          LABELS "host;sm87;prefill;mlp;a4w4;k512;gateup;fragment-native;m64;n128;one-cta;contract"
        )

        add_executable(
          q3x_sm87_a4w4_gateup_k512_fragment_native_m64n128_1cta_correctness_test
          tests/sm87_a4w4_gateup_k512_fragment_native_m64n128_1cta_correctness_test.cu
        )
        target_link_libraries(
          q3x_sm87_a4w4_gateup_k512_fragment_native_m64n128_1cta_correctness_test
          PRIVATE q3x::kernels CUDA::cudart
        )
        q3x_enable_warnings(
          q3x_sm87_a4w4_gateup_k512_fragment_native_m64n128_1cta_correctness_test
        )
      endif()

      if(Q3X_BUILD_SM87_A4W4_GATEUP_K512_FRAGMENT_NATIVE_M128N64_STAGED_EXPERIMENT)
        add_executable(
          q3x_sm87_a4w4_gateup_k512_fragment_native_m128n64_staged_contract_test
          tests/sm87_a4w4_gateup_k512_fragment_native_m128n64_staged_contract_test.cpp
        )
        target_link_libraries(
          q3x_sm87_a4w4_gateup_k512_fragment_native_m128n64_staged_contract_test
          PRIVATE q3x::kernels CUDA::cudart
        )
        q3x_enable_warnings(
          q3x_sm87_a4w4_gateup_k512_fragment_native_m128n64_staged_contract_test
        )
        add_test(NAME sm87_a4w4_gateup_k512_fragment_native_m128n64_staged_contract
          COMMAND q3x_sm87_a4w4_gateup_k512_fragment_native_m128n64_staged_contract_test
        )
        set_tests_properties(
          sm87_a4w4_gateup_k512_fragment_native_m128n64_staged_contract PROPERTIES
          LABELS "host;sm87;prefill;mlp;a4w4;k512;gateup;fragment-native;m128;n64;staged;contract"
        )

        add_executable(
          q3x_sm87_a4w4_gateup_k512_fragment_native_m128n64_staged_correctness_test
          tests/sm87_a4w4_gateup_k512_fragment_native_m128n64_staged_correctness_test.cu
        )
        target_link_libraries(
          q3x_sm87_a4w4_gateup_k512_fragment_native_m128n64_staged_correctness_test
          PRIVATE q3x::kernels CUDA::cudart
        )
        q3x_enable_warnings(
          q3x_sm87_a4w4_gateup_k512_fragment_native_m128n64_staged_correctness_test
        )
      endif()

      if(Q3X_BUILD_SM87_A4W4_DOWN_K512_FRAGMENT_NATIVE_M128N256_1CTA_EXPERIMENT)
        add_executable(
          q3x_sm87_a4w4_down_k512_fragment_native_m128n256_1cta_contract_test
          tests/sm87_a4w4_down_k512_fragment_native_m128n256_1cta_contract_test.cpp
        )
        target_link_libraries(
          q3x_sm87_a4w4_down_k512_fragment_native_m128n256_1cta_contract_test
          PRIVATE q3x::kernels CUDA::cudart
        )
        q3x_enable_warnings(
          q3x_sm87_a4w4_down_k512_fragment_native_m128n256_1cta_contract_test
        )
        add_test(NAME sm87_a4w4_down_k512_fragment_native_m128n256_1cta_contract
          COMMAND q3x_sm87_a4w4_down_k512_fragment_native_m128n256_1cta_contract_test
        )
        set_tests_properties(
          sm87_a4w4_down_k512_fragment_native_m128n256_1cta_contract PROPERTIES
          LABELS "host;sm87;prefill;mlp;a4w4;k512;down;fragment-native;m128;n256;one-cta;contract"
        )

        add_executable(
          q3x_sm87_a4w4_down_k512_fragment_native_m128n256_1cta_correctness_test
          tests/sm87_a4w4_down_k512_fragment_native_m128n256_1cta_correctness_test.cu
        )
        target_link_libraries(
          q3x_sm87_a4w4_down_k512_fragment_native_m128n256_1cta_correctness_test
          PRIVATE q3x::kernels CUDA::cudart
        )
        q3x_enable_warnings(
          q3x_sm87_a4w4_down_k512_fragment_native_m128n256_1cta_correctness_test
        )
      endif()

      add_executable(q3x_sm87_a4w4_down_k512_fragment_native_contract_test
        tests/sm87_a4w4_down_k512_fragment_native_contract_test.cpp
      )
      target_link_libraries(
        q3x_sm87_a4w4_down_k512_fragment_native_contract_test
        PRIVATE q3x::kernels CUDA::cudart
      )
      q3x_enable_warnings(
        q3x_sm87_a4w4_down_k512_fragment_native_contract_test
      )
      add_test(NAME sm87_a4w4_down_k512_fragment_native_contract
        COMMAND q3x_sm87_a4w4_down_k512_fragment_native_contract_test
      )
      set_tests_properties(
        sm87_a4w4_down_k512_fragment_native_contract PROPERTIES
        LABELS "host;sm87;prefill;mlp;a4w4;k512;down;fragment-native;contract"
      )

      add_executable(q3x_sm87_a4w4_down_k512_fragment_native_correctness_test
        tests/sm87_a4w4_down_k512_fragment_native_correctness_test.cu
      )
      target_link_libraries(
        q3x_sm87_a4w4_down_k512_fragment_native_correctness_test
        PRIVATE q3x::kernels CUDA::cudart
      )
      q3x_enable_warnings(
        q3x_sm87_a4w4_down_k512_fragment_native_correctness_test
      )
    endif()

    find_program(Q3X_FLOCK_EXECUTABLE flock REQUIRED)
    add_test(NAME sm87_a4_quantize_bf16_k512_split_correctness
      COMMAND "${Q3X_FLOCK_EXECUTABLE}" -x /tmp/q3x-gpu-bench.lock
              $<TARGET_FILE:q3x_sm87_a4_quantize_bf16_k512_split_correctness_test>
    )
    set_tests_properties(
      sm87_a4_quantize_bf16_k512_split_correctness PROPERTIES
      LABELS "cuda;sm87;prefill;mlp;a4w4;k512;quantization;correctness;synthetic"
      SKIP_RETURN_CODE 77
      RESOURCE_LOCK q3x_sm87_gpu
      TIMEOUT 300
    )

    add_test(NAME sm87_a4w4_gateup_k512_macrocell_correctness
      COMMAND "${Q3X_FLOCK_EXECUTABLE}" -x /tmp/q3x-gpu-bench.lock
              $<TARGET_FILE:q3x_sm87_a4w4_gateup_k512_macrocell_correctness_test>
    )
    set_tests_properties(
      sm87_a4w4_gateup_k512_macrocell_correctness PROPERTIES
      LABELS "cuda;sm87;prefill;mlp;a4w4;k512;gateup;correctness;synthetic"
      SKIP_RETURN_CODE 77
      RESOURCE_LOCK q3x_sm87_gpu
      TIMEOUT 300
    )
    if(Q3X_BUILD_SM87_A4W4_GATEUP_K512_M128N128_PROJECTION_SERIAL_EXPERIMENT)
      add_test(
        NAME sm87_a4w4_gateup_k512_m128n128_projection_serial_correctness
        COMMAND "${Q3X_FLOCK_EXECUTABLE}" -x /tmp/q3x-gpu-bench.lock
                $<TARGET_FILE:q3x_sm87_a4w4_gateup_k512_m128n128_projection_serial_correctness_test>
      )
      set_tests_properties(
        sm87_a4w4_gateup_k512_m128n128_projection_serial_correctness
        PROPERTIES
        LABELS "cuda;sm87;prefill;mlp;a4w4;k512;gateup;m128n128;projection-serial;correctness;synthetic"
        SKIP_RETURN_CODE 77
        RESOURCE_LOCK q3x_sm87_gpu
        TIMEOUT 300
      )
    endif()
    if(Q3X_BUILD_SM87_A4W4_GATEUP_K512_M128N512_FUSED_QUANTIZE_EXPERIMENT)
      add_test(
        NAME sm87_a4w4_gateup_k512_m128n512_fused_quantize_resources
        COMMAND "${Q3X_FLOCK_EXECUTABLE}" -x /tmp/q3x-gpu-bench.lock
                $<TARGET_FILE:q3x_sm87_a4w4_gateup_k512_m128n512_fused_quantize_resource_test>
      )
      set_tests_properties(
        sm87_a4w4_gateup_k512_m128n512_fused_quantize_resources
        PROPERTIES
        LABELS "cuda;sm87;prefill;mlp;a4w4;k512;gateup;m128n512;fused-quantize;resources"
        SKIP_RETURN_CODE 77
        RESOURCE_LOCK q3x_sm87_gpu
        TIMEOUT 300
      )
      add_test(
        NAME sm87_a4w4_gateup_k512_m128n512_fused_quantize_correctness
        COMMAND "${Q3X_FLOCK_EXECUTABLE}" -x /tmp/q3x-gpu-bench.lock
                $<TARGET_FILE:q3x_sm87_a4w4_gateup_k512_m128n512_fused_quantize_correctness_test>
      )
      set_tests_properties(
        sm87_a4w4_gateup_k512_m128n512_fused_quantize_correctness
        PROPERTIES
        LABELS "cuda;sm87;prefill;mlp;a4w4;k512;gateup;m128n512;fused-quantize;correctness;scalar;quantization;canary;graph;synthetic"
        SKIP_RETURN_CODE 77
        RESOURCE_LOCK q3x_sm87_gpu
        TIMEOUT 600
      )
    endif()
    add_test(NAME sm87_a4w4_down_k512_macrocell_correctness
      COMMAND "${Q3X_FLOCK_EXECUTABLE}" -x /tmp/q3x-gpu-bench.lock
              $<TARGET_FILE:q3x_sm87_a4w4_down_k512_macrocell_correctness_test>
    )
    set_tests_properties(
      sm87_a4w4_down_k512_macrocell_correctness PROPERTIES
      LABELS "cuda;sm87;prefill;mlp;a4w4;k512;down;correctness;synthetic"
      SKIP_RETURN_CODE 77
      RESOURCE_LOCK q3x_sm87_gpu
      TIMEOUT 300
    )
    add_test(NAME sm87_a4w4_gateup_down_k512_edge_correctness
      COMMAND "${Q3X_FLOCK_EXECUTABLE}" -x /tmp/q3x-gpu-bench.lock
              $<TARGET_FILE:q3x_sm87_a4w4_gateup_down_k512_edge_correctness_test>
    )
    set_tests_properties(
      sm87_a4w4_gateup_down_k512_edge_correctness PROPERTIES
      LABELS "cuda;sm87;prefill;mlp;a4w4;k512;gateup;down;edge;correctness;resources;synthetic"
      SKIP_RETURN_CODE 77
      RESOURCE_LOCK q3x_sm87_gpu
      TIMEOUT 300
    )
    if(Q3X_BUILD_SM87_A4W4_GATEUP_DOWN_K512_EDGE_M64N128_K256_ALTERNATING_EXPERIMENT)
      add_test(
        NAME sm87_a4w4_gateup_down_k512_edge_m64n128_k256_alternating_correctness
        COMMAND "${Q3X_FLOCK_EXECUTABLE}" -x /tmp/q3x-gpu-bench.lock
                $<TARGET_FILE:q3x_sm87_a4w4_gateup_down_k512_edge_m64n128_k256_alternating_correctness_test>
      )
      set_tests_properties(
        sm87_a4w4_gateup_down_k512_edge_m64n128_k256_alternating_correctness
        PROPERTIES
        LABELS "cuda;sm87;prefill;mlp;a4w4;k512;gateup;down;edge;alternating;k256;correctness;resources;graph;synthetic"
        SKIP_RETURN_CODE 77
        RESOURCE_LOCK q3x_sm87_gpu
        TIMEOUT 300
      )
      add_test(
        NAME sm87_a4w4_gateup_down_k512_edge_m64n128_k256_ldmatrix_pairfeed_correctness
        COMMAND "${Q3X_FLOCK_EXECUTABLE}" -x /tmp/q3x-gpu-bench.lock
                $<TARGET_FILE:q3x_sm87_a4w4_gateup_down_k512_edge_m64n128_k256_ldmatrix_pairfeed_correctness_test>
      )
      set_tests_properties(
        sm87_a4w4_gateup_down_k512_edge_m64n128_k256_ldmatrix_pairfeed_correctness
        PROPERTIES
        LABELS "cuda;sm87;prefill;mlp;a4w4;k512;gateup;down;edge;ldmatrix;pairfeed;correctness;resources;graph;synthetic"
        SKIP_RETURN_CODE 77
        RESOURCE_LOCK q3x_sm87_gpu
        TIMEOUT 300
      )
    endif()
    if(Q3X_BUILD_SM87_A4W4_GATEUP_DOWN_K512_EDGE_M128N512_LDMATRIX_ADMISSION)
      add_test(
        NAME sm87_a4w4_gateup_down_k512_edge_m128n512_ldmatrix_correctness
        COMMAND "${Q3X_FLOCK_EXECUTABLE}" -x /tmp/q3x-gpu-bench.lock
                $<TARGET_FILE:q3x_sm87_a4w4_gateup_down_k512_edge_m128n512_ldmatrix_correctness_test>
      )
      set_tests_properties(
        sm87_a4w4_gateup_down_k512_edge_m128n512_ldmatrix_correctness
        PROPERTIES
        LABELS "cuda;sm87;prefill;mlp;a4w4;k512;gateup;down;edge;m128n512;ldmatrix;correctness;resources;graph;scratch;synthetic"
        SKIP_RETURN_CODE 77
        RESOURCE_LOCK q3x_sm87_gpu
        TIMEOUT 300
      )
    endif()
    if(Q3X_BUILD_SM87_A4W4_GATEUP_DOWN_K512_EDGE_M128N512_PAIRED_LDMATRIX_ADMISSION)
      add_test(
        NAME sm87_a4w4_gateup_down_k512_edge_m128n512_paired_ldmatrix_correctness
        COMMAND "${Q3X_FLOCK_EXECUTABLE}" -x /tmp/q3x-gpu-bench.lock
                $<TARGET_FILE:q3x_sm87_a4w4_gateup_down_k512_edge_m128n512_paired_ldmatrix_correctness_test>
      )
      set_tests_properties(
        sm87_a4w4_gateup_down_k512_edge_m128n512_paired_ldmatrix_correctness
        PROPERTIES
        LABELS "cuda;sm87;prefill;mlp;a4w4;k512;gateup;down;edge;m128n512;paired;ldmatrix;correctness;resources;graph;scratch;synthetic"
        SKIP_RETURN_CODE 77
        RESOURCE_LOCK q3x_sm87_gpu
        TIMEOUT 300
      )
    endif()
    if(Q3X_BUILD_SM87_A4W4_DOWN_K512_M128N128_LDMATRIX_PAIRRING_ADMISSION)
      add_test(
        NAME sm87_a4w4_down_k512_m128n128_ldmatrix_pairring_correctness
        COMMAND "${Q3X_FLOCK_EXECUTABLE}" -x /tmp/q3x-gpu-bench.lock
                $<TARGET_FILE:q3x_sm87_a4w4_down_k512_m128n128_ldmatrix_pairring_correctness_test>
      )
      set_tests_properties(
        sm87_a4w4_down_k512_m128n128_ldmatrix_pairring_correctness
        PROPERTIES
        LABELS "cuda;sm87;prefill;mlp;a4w4;k512;down;m128n128;ldmatrix;pairring;correctness;resources;graph;synthetic"
        SKIP_RETURN_CODE 77
        RESOURCE_LOCK q3x_sm87_gpu
        TIMEOUT 300
      )
    endif()
    if(Q3X_BUILD_SM87_A4W4_DOWN_K512_M128N128_16WARP_PAIRRING_EXPERIMENT AND
       Q3X_BUILD_SM87_A4W4_DOWN_K512_M128N128_LDMATRIX_PAIRRING_ADMISSION)
      add_test(
        NAME sm87_a4w4_down_k512_m128n128_16warp_pairring_correctness
        COMMAND "${Q3X_FLOCK_EXECUTABLE}" -x /tmp/q3x-gpu-bench.lock
                $<TARGET_FILE:q3x_sm87_a4w4_down_k512_m128n128_16warp_pairring_correctness_test>
      )
      set_tests_properties(
        sm87_a4w4_down_k512_m128n128_16warp_pairring_correctness
        PROPERTIES
        LABELS "cuda;sm87;prefill;mlp;a4w4;k512;down;m128n128;16warp;pairring;correctness;resources;graph;synthetic"
        SKIP_RETURN_CODE 77
        RESOURCE_LOCK q3x_sm87_gpu
        TIMEOUT 600
      )
    endif()
    if(Q3X_BUILD_SM87_A4W4_GATEUP_DOWN_K512_EDGE_M128N64_EXPERIMENT)
      add_test(NAME sm87_a4w4_gateup_down_k512_edge_m128n64_correctness
        COMMAND "${Q3X_FLOCK_EXECUTABLE}" -x /tmp/q3x-gpu-bench.lock
                $<TARGET_FILE:q3x_sm87_a4w4_gateup_down_k512_edge_m128n64_correctness_test>
      )
      set_tests_properties(
        sm87_a4w4_gateup_down_k512_edge_m128n64_correctness PROPERTIES
        LABELS "cuda;sm87;prefill;mlp;a4w4;k512;gateup;down;edge;m128n64;correctness;resources;synthetic"
        SKIP_RETURN_CODE 77
        RESOURCE_LOCK q3x_sm87_gpu
        TIMEOUT 300
      )
    endif()
    if(Q3X_BUILD_SM87_A4W4_DOWN_K512_M16N64_V2_EXPERIMENT)
      add_test(NAME sm87_a4w4_down_k512_m16n64_v2_correctness
        COMMAND "${Q3X_FLOCK_EXECUTABLE}" -x /tmp/q3x-gpu-bench.lock
                $<TARGET_FILE:q3x_sm87_a4w4_down_k512_m16n64_v2_correctness_test>
      )
      set_tests_properties(
        sm87_a4w4_down_k512_m16n64_v2_correctness PROPERTIES
        LABELS "cuda;sm87;prefill;mlp;a4w4;k512;down;v2;correctness;resources;synthetic"
        SKIP_RETURN_CODE 77
        RESOURCE_LOCK q3x_sm87_gpu
        TIMEOUT 300
      )
    endif()
    if(Q3X_BUILD_SM87_A4W4_MLP_K512_GLOBAL_FLOW_EXPERIMENT)
      add_test(NAME sm87_a4w4_gateup_k512_m128n64_global_flow_correctness
        COMMAND "${Q3X_FLOCK_EXECUTABLE}" -x /tmp/q3x-gpu-bench.lock
                $<TARGET_FILE:q3x_sm87_a4w4_gateup_k512_m128n64_global_flow_correctness_test>
      )
      set_tests_properties(
        sm87_a4w4_gateup_k512_m128n64_global_flow_correctness PROPERTIES
        LABELS "cuda;sm87;prefill;mlp;a4w4;k512;gateup;global-flow;correctness;synthetic"
        SKIP_RETURN_CODE 77
        RESOURCE_LOCK q3x_sm87_gpu
        TIMEOUT 300
      )
      add_test(NAME sm87_a4w4_down_k512_bstationary_correctness
        COMMAND "${Q3X_FLOCK_EXECUTABLE}" -x /tmp/q3x-gpu-bench.lock
                $<TARGET_FILE:q3x_sm87_a4w4_down_k512_bstationary_correctness_test>
      )
      set_tests_properties(
        sm87_a4w4_down_k512_bstationary_correctness PROPERTIES
        LABELS "cuda;sm87;prefill;mlp;a4w4;k512;down;bstationary;correctness;synthetic"
        SKIP_RETURN_CODE 77
        RESOURCE_LOCK q3x_sm87_gpu
        TIMEOUT 300
      )
    endif()
    if(Q3X_BUILD_SM87_A4W4_MLP_K512_FRAGMENT_NATIVE_EXPERIMENT)
      add_test(NAME sm87_a4w4_gateup_k512_fragment_native_correctness
        COMMAND "${Q3X_FLOCK_EXECUTABLE}" -x /tmp/q3x-gpu-bench.lock
                $<TARGET_FILE:q3x_sm87_a4w4_gateup_k512_fragment_native_correctness_test>
      )
      set_tests_properties(
        sm87_a4w4_gateup_k512_fragment_native_correctness PROPERTIES
        LABELS "cuda;sm87;prefill;mlp;a4w4;k512;gateup;fragment-native;correctness;synthetic"
        SKIP_RETURN_CODE 77
        RESOURCE_LOCK q3x_sm87_gpu
        TIMEOUT 300
      )
      if(Q3X_BUILD_SM87_A4W4_GATEUP_K512_FRAGMENT_NATIVE_M128_EXPERIMENT)
        add_test(NAME sm87_a4w4_gateup_k512_fragment_native_m128_correctness
          COMMAND "${Q3X_FLOCK_EXECUTABLE}" -x /tmp/q3x-gpu-bench.lock
                  $<TARGET_FILE:q3x_sm87_a4w4_gateup_k512_fragment_native_m128_correctness_test>
        )
        set_tests_properties(
          sm87_a4w4_gateup_k512_fragment_native_m128_correctness PROPERTIES
          LABELS "cuda;sm87;prefill;mlp;a4w4;k512;gateup;fragment-native;m128;correctness;resources;synthetic"
          SKIP_RETURN_CODE 77
          RESOURCE_LOCK q3x_sm87_gpu
          TIMEOUT 300
        )
      endif()
      if(Q3X_BUILD_SM87_A4W4_GATEUP_K512_FRAGMENT_NATIVE_M128N64_1CTA_EXPERIMENT)
        add_test(NAME sm87_a4w4_gateup_k512_fragment_native_m128n64_1cta_correctness
          COMMAND "${Q3X_FLOCK_EXECUTABLE}" -x /tmp/q3x-gpu-bench.lock
                  $<TARGET_FILE:q3x_sm87_a4w4_gateup_k512_fragment_native_m128n64_1cta_correctness_test>
        )
        set_tests_properties(
          sm87_a4w4_gateup_k512_fragment_native_m128n64_1cta_correctness PROPERTIES
          LABELS "cuda;sm87;prefill;mlp;a4w4;k512;gateup;fragment-native;m128;n64;one-cta;correctness;resources;synthetic"
          SKIP_RETURN_CODE 77
          RESOURCE_LOCK q3x_sm87_gpu
          TIMEOUT 300
        )
      endif()
      if(Q3X_BUILD_SM87_A4W4_GATEUP_K512_FRAGMENT_NATIVE_M64N128_1CTA_EXPERIMENT)
        add_test(NAME sm87_a4w4_gateup_k512_fragment_native_m64n128_1cta_correctness
          COMMAND "${Q3X_FLOCK_EXECUTABLE}" -x /tmp/q3x-gpu-bench.lock
                  $<TARGET_FILE:q3x_sm87_a4w4_gateup_k512_fragment_native_m64n128_1cta_correctness_test>
        )
        set_tests_properties(
          sm87_a4w4_gateup_k512_fragment_native_m64n128_1cta_correctness PROPERTIES
          LABELS "cuda;sm87;prefill;mlp;a4w4;k512;gateup;fragment-native;m64;n128;one-cta;correctness;resources;synthetic"
          SKIP_RETURN_CODE 77
          RESOURCE_LOCK q3x_sm87_gpu
          TIMEOUT 300
        )
      endif()
      if(Q3X_BUILD_SM87_A4W4_GATEUP_K512_FRAGMENT_NATIVE_M128N64_STAGED_EXPERIMENT)
        add_test(NAME sm87_a4w4_gateup_k512_fragment_native_m128n64_staged_correctness
          COMMAND "${Q3X_FLOCK_EXECUTABLE}" -x /tmp/q3x-gpu-bench.lock
                  $<TARGET_FILE:q3x_sm87_a4w4_gateup_k512_fragment_native_m128n64_staged_correctness_test>
        )
        set_tests_properties(
          sm87_a4w4_gateup_k512_fragment_native_m128n64_staged_correctness PROPERTIES
          LABELS "cuda;sm87;prefill;mlp;a4w4;k512;gateup;fragment-native;m128;n64;staged;correctness;resources;synthetic"
          SKIP_RETURN_CODE 77
          RESOURCE_LOCK q3x_sm87_gpu
          TIMEOUT 300
        )
      endif()
      if(Q3X_BUILD_SM87_A4W4_DOWN_K512_FRAGMENT_NATIVE_M128N256_1CTA_EXPERIMENT)
        add_test(NAME sm87_a4w4_down_k512_fragment_native_m128n256_1cta_correctness
          COMMAND "${Q3X_FLOCK_EXECUTABLE}" -x /tmp/q3x-gpu-bench.lock
                  $<TARGET_FILE:q3x_sm87_a4w4_down_k512_fragment_native_m128n256_1cta_correctness_test>
        )
        set_tests_properties(
          sm87_a4w4_down_k512_fragment_native_m128n256_1cta_correctness PROPERTIES
          LABELS "cuda;sm87;prefill;mlp;a4w4;k512;down;fragment-native;m128;n256;one-cta;correctness;resources;synthetic"
          SKIP_RETURN_CODE 77
          RESOURCE_LOCK q3x_sm87_gpu
          TIMEOUT 300
        )
      endif()
      add_test(NAME sm87_a4w4_down_k512_fragment_native_correctness
        COMMAND "${Q3X_FLOCK_EXECUTABLE}" -x /tmp/q3x-gpu-bench.lock
                $<TARGET_FILE:q3x_sm87_a4w4_down_k512_fragment_native_correctness_test>
      )
      set_tests_properties(
        sm87_a4w4_down_k512_fragment_native_correctness PROPERTIES
        LABELS "cuda;sm87;prefill;mlp;a4w4;k512;down;fragment-native;correctness;synthetic"
        SKIP_RETURN_CODE 77
        RESOURCE_LOCK q3x_sm87_gpu
        TIMEOUT 300
      )
    endif()
  endif()

  if(Q3X_BUILD_A4W4_FULL_PREFILL_ADMISSION)
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
      LABELS "cuda;sm87;prefill;gdn;correctness;synthetic;component"
      SKIP_RETURN_CODE 77
      RESOURCE_LOCK q3x_sm87_gpu
    )
  endif()

  # The synthetic correctness/resource target can own the candidate TU in an
  # isolated build or consume the same production-linked TU in an admission
  # build, but never compiles a duplicate definition into one executable.
  if(Q3X_BUILD_GDN_PREFILL_PROMPT_SPAN_MACRO_EXPERIMENT)
    add_executable(q3x_gdn_prefill_prompt_span_macro_c64_component_test
      tests/gdn_prefill_prompt_span_macro_c64_component_test.cu
    )
    if(NOT Q3X_BUILD_GDN_PREFILL_PROMPT_SPAN_MACRO_ADMISSION)
      target_sources(q3x_gdn_prefill_prompt_span_macro_c64_component_test
        PRIVATE src/kernels/sm87/gdn_prefill_prompt_span_macro_sm87.cu
      )
    endif()
    target_compile_features(
      q3x_gdn_prefill_prompt_span_macro_c64_component_test
      PRIVATE cxx_std_17 cuda_std_17
    )
    target_include_directories(
      q3x_gdn_prefill_prompt_span_macro_c64_component_test
      PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/sm87"
    )
    target_link_libraries(
      q3x_gdn_prefill_prompt_span_macro_c64_component_test
      PRIVATE q3x::kernels CUDA::cudart
    )
    set_target_properties(
      q3x_gdn_prefill_prompt_span_macro_c64_component_test
      PROPERTIES CUDA_ARCHITECTURES 87
    )
    target_compile_options(
      q3x_gdn_prefill_prompt_span_macro_c64_component_test PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-Xptxas=-v;-Xptxas=-warn-spills>
    )
    q3x_enable_warnings(
      q3x_gdn_prefill_prompt_span_macro_c64_component_test
    )
    add_test(NAME gdn_prefill_prompt_span_macro_c64_component
      COMMAND $<TARGET_FILE:q3x_gdn_prefill_prompt_span_macro_c64_component_test>
    )
    set_tests_properties(
      gdn_prefill_prompt_span_macro_c64_component PROPERTIES
      LABELS "synthetic;correctness;resources;gdn;prefill;sm87;cuda"
      SKIP_RETURN_CODE 77
      RESOURCE_LOCK q3x_sm87_gpu
      TIMEOUT 300
    )
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
  if(Q3X_BUILD_A4W4_FULL_PREFILL_ADMISSION)
    add_test(NAME reference_runner_host_gateup_complete_cell_v2_admission
      COMMAND q3x_reference_runner_host_test
    )
    set_tests_properties(
      reference_runner_host_gateup_complete_cell_v2_admission PROPERTIES
      ENVIRONMENT
        "Q3X_DISABLE_OPTIMIZED_PREFILL=0;Q3X_RUN_A4W4_GATEUP_COMPLETE_CELL_V2_ADMISSION=1;Q3X_RUN_A4W4_M128_STAGE_MAJOR_ADMISSION=0;Q3X_RUN_A4W4_DOWN_M128_STAGE_MAJOR_ADMISSION=0"
      LABELS "host;sm87;prefill;a4w4;k128;admission;environment"
    )
  endif()
  if(Q3X_BUILD_SM87_A4W4_GATEUP_PROJECTION_V3_ADMISSION)
    add_test(NAME reference_runner_host_gateup_projection_v3_admission
      COMMAND q3x_reference_runner_host_test
    )
    set_tests_properties(
      reference_runner_host_gateup_projection_v3_admission PROPERTIES
      ENVIRONMENT
        "Q3X_DISABLE_OPTIMIZED_PREFILL=0;Q3X_RUN_A4W4_GATEUP_PROJECTION_V3_ADMISSION=1;Q3X_RUN_A4W4_GATEUP_COMPLETE_CELL_V2_ADMISSION=1;Q3X_RUN_A4W4_M128_STAGE_MAJOR_ADMISSION=1"
      LABELS "host;sm87;prefill;a4w4;k128;gateup;v3;admission;environment"
    )
    add_test(NAME reference_runner_host_gateup_projection_v3_nonexact_env
      COMMAND q3x_reference_runner_host_test
    )
    set_tests_properties(
      reference_runner_host_gateup_projection_v3_nonexact_env PROPERTIES
      ENVIRONMENT
        "Q3X_DISABLE_OPTIMIZED_PREFILL=0;Q3X_RUN_A4W4_GATEUP_PROJECTION_V3_ADMISSION=true"
      LABELS "host;sm87;prefill;a4w4;k128;gateup;v3;admission;environment;negative"
    )
    add_test(NAME reference_runner_host_gateup_projection_v3_global_disable
      COMMAND q3x_reference_runner_host_test
    )
    set_tests_properties(
      reference_runner_host_gateup_projection_v3_global_disable PROPERTIES
      ENVIRONMENT
        "Q3X_DISABLE_OPTIMIZED_PREFILL=1;Q3X_RUN_A4W4_GATEUP_PROJECTION_V3_ADMISSION=1"
      LABELS "host;sm87;prefill;a4w4;k128;gateup;v3;admission;environment;negative"
    )
  endif()
  if(Q3X_BUILD_SM87_A4W4_ATTENTION_SUPERMATRIX_ADMISSION)
    add_test(NAME reference_runner_host_attention_supermatrix_admission
      COMMAND q3x_reference_runner_host_test
    )
    set_tests_properties(
      reference_runner_host_attention_supermatrix_admission PROPERTIES
      ENVIRONMENT
        "Q3X_DISABLE_OPTIMIZED_PREFILL=0;Q3X_RUN_A4W4_ATTENTION_SUPERMATRIX_ADMISSION=1;Q3X_RUN_A4W4_M128_STAGE_MAJOR_ADMISSION=1"
      LABELS "host;sm87;prefill;a4w4;k128;attention;supermatrix;admission;environment"
    )
    add_test(NAME reference_runner_host_attention_supermatrix_nonexact_env
      COMMAND q3x_reference_runner_host_test
    )
    set_tests_properties(
      reference_runner_host_attention_supermatrix_nonexact_env PROPERTIES
      ENVIRONMENT
        "Q3X_DISABLE_OPTIMIZED_PREFILL=0;Q3X_RUN_A4W4_ATTENTION_SUPERMATRIX_ADMISSION=true"
      LABELS "host;sm87;prefill;a4w4;k128;attention;supermatrix;admission;environment;negative"
    )
    add_test(NAME reference_runner_host_attention_supermatrix_global_disable
      COMMAND q3x_reference_runner_host_test
    )
    set_tests_properties(
      reference_runner_host_attention_supermatrix_global_disable PROPERTIES
      ENVIRONMENT
        "Q3X_DISABLE_OPTIMIZED_PREFILL=1;Q3X_RUN_A4W4_ATTENTION_SUPERMATRIX_ADMISSION=1"
      LABELS "host;sm87;prefill;a4w4;k128;attention;supermatrix;admission;environment;negative"
    )
  endif()
  if(Q3X_BUILD_SM87_A4W4_ATTENTION_K256_M128N256_A_EXCHANGE_B4_EXPERIMENT)
    add_test(
      NAME reference_runner_host_attention_k256_a_exchange_b4_admission
      COMMAND q3x_reference_runner_host_test
    )
    set_tests_properties(
      reference_runner_host_attention_k256_a_exchange_b4_admission PROPERTIES
      ENVIRONMENT
        "Q3X_DISABLE_OPTIMIZED_PREFILL=0;Q3X_RUN_A4W4_ATTENTION_K256_M128N256_ADMISSION=0;Q3X_RUN_A4W4_ATTENTION_K256_M128N256_A_EXCHANGE_B4_ADMISSION=1"
      LABELS "host;sm87;prefill;a4w4;k256;attention;a_exchange_b4;admission;environment"
    )
    add_test(
      NAME reference_runner_host_attention_k256_a_exchange_b4_nonexact_env
      COMMAND q3x_reference_runner_host_test
    )
    set_tests_properties(
      reference_runner_host_attention_k256_a_exchange_b4_nonexact_env
      PROPERTIES
      ENVIRONMENT
        "Q3X_DISABLE_OPTIMIZED_PREFILL=0;Q3X_RUN_A4W4_ATTENTION_K256_M128N256_ADMISSION=0;Q3X_RUN_A4W4_ATTENTION_K256_M128N256_A_EXCHANGE_B4_ADMISSION=true"
      LABELS "host;sm87;prefill;a4w4;k256;attention;a_exchange_b4;admission;environment;negative"
    )
    add_test(
      NAME reference_runner_host_attention_k256_a_exchange_b4_global_disable
      COMMAND q3x_reference_runner_host_test
    )
    set_tests_properties(
      reference_runner_host_attention_k256_a_exchange_b4_global_disable
      PROPERTIES
      ENVIRONMENT
        "Q3X_DISABLE_OPTIMIZED_PREFILL=1;Q3X_RUN_A4W4_ATTENTION_K256_M128N256_ADMISSION=0;Q3X_RUN_A4W4_ATTENTION_K256_M128N256_A_EXCHANGE_B4_ADMISSION=1"
      LABELS "host;sm87;prefill;a4w4;k256;attention;a_exchange_b4;admission;environment;negative"
    )
  endif()
  if(Q3X_BUILD_SM87_A4W4_DOWN_COMPLETE_CELL_V2_ADMISSION)
    add_test(NAME reference_runner_host_down_complete_cell_v2_admission
      COMMAND q3x_reference_runner_host_test
    )
    set_tests_properties(
      reference_runner_host_down_complete_cell_v2_admission PROPERTIES
      ENVIRONMENT
        "Q3X_DISABLE_OPTIMIZED_PREFILL=0;Q3X_RUN_A4W4_DOWN_COMPLETE_CELL_V2_ADMISSION=1;Q3X_RUN_A4W4_DOWN_M128_STAGE_MAJOR_ADMISSION=1"
      LABELS "host;sm87;prefill;a4w4;k128;down;complete_cell;admission;environment"
    )
    add_test(NAME reference_runner_host_down_complete_cell_v2_nonexact_env
      COMMAND q3x_reference_runner_host_test
    )
    set_tests_properties(
      reference_runner_host_down_complete_cell_v2_nonexact_env PROPERTIES
      ENVIRONMENT
        "Q3X_DISABLE_OPTIMIZED_PREFILL=0;Q3X_RUN_A4W4_DOWN_COMPLETE_CELL_V2_ADMISSION=true"
      LABELS "host;sm87;prefill;a4w4;k128;down;complete_cell;admission;environment;negative"
    )
    add_test(NAME reference_runner_host_down_complete_cell_v2_global_disable
      COMMAND q3x_reference_runner_host_test
    )
    set_tests_properties(
      reference_runner_host_down_complete_cell_v2_global_disable PROPERTIES
      ENVIRONMENT
        "Q3X_DISABLE_OPTIMIZED_PREFILL=1;Q3X_RUN_A4W4_DOWN_COMPLETE_CELL_V2_ADMISSION=1"
      LABELS "host;sm87;prefill;a4w4;k128;down;complete_cell;admission;environment;negative"
    )
  endif()
  if(Q3X_BUILD_SM87_A4W4_DOWN_COMPLETE_CELL_V3_ADMISSION)
    add_test(NAME reference_runner_host_down_complete_cell_v3_admission
      COMMAND q3x_reference_runner_host_test
    )
    set_tests_properties(
      reference_runner_host_down_complete_cell_v3_admission PROPERTIES
      ENVIRONMENT
        "Q3X_DISABLE_OPTIMIZED_PREFILL=0;Q3X_RUN_A4W4_DOWN_COMPLETE_CELL_V3_ADMISSION=1;Q3X_RUN_A4W4_DOWN_COMPLETE_CELL_V2_ADMISSION=1;Q3X_RUN_A4W4_DOWN_M128_STAGE_MAJOR_ADMISSION=1"
      LABELS "host;sm87;prefill;a4w4;k128;down;complete_cell;v3;admission;environment"
    )
    add_test(NAME reference_runner_host_down_complete_cell_v3_nonexact_env
      COMMAND q3x_reference_runner_host_test
    )
    set_tests_properties(
      reference_runner_host_down_complete_cell_v3_nonexact_env PROPERTIES
      ENVIRONMENT
        "Q3X_DISABLE_OPTIMIZED_PREFILL=0;Q3X_RUN_A4W4_DOWN_COMPLETE_CELL_V3_ADMISSION=true"
      LABELS "host;sm87;prefill;a4w4;k128;down;complete_cell;v3;admission;environment;negative"
    )
    add_test(NAME reference_runner_host_down_complete_cell_v3_global_disable
      COMMAND q3x_reference_runner_host_test
    )
    set_tests_properties(
      reference_runner_host_down_complete_cell_v3_global_disable PROPERTIES
      ENVIRONMENT
        "Q3X_DISABLE_OPTIMIZED_PREFILL=1;Q3X_RUN_A4W4_DOWN_COMPLETE_CELL_V3_ADMISSION=1"
      LABELS "host;sm87;prefill;a4w4;k128;down;complete_cell;v3;admission;environment;negative"
    )
  endif()
  if(Q3X_BUILD_FULL_ATTENTION_LONG_CONTEXT_GROUP_Q64_ADMISSION)
    add_test(NAME reference_runner_host_long_context_group_q64_admission
      COMMAND q3x_reference_runner_host_test
    )
    set_tests_properties(
      reference_runner_host_long_context_group_q64_admission PROPERTIES
      ENVIRONMENT
        "Q3X_RUN_FULL_ATTENTION_LONG_CONTEXT_GROUP_Q64_ADMISSION=1"
      LABELS "host;sm87;prefill;attention;admission"
    )
  endif()

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

  add_executable(q3x_openai_protocol_test
    tests/openai_protocol_test.cpp
  )
  target_link_libraries(q3x_openai_protocol_test
    PRIVATE q3x_eval_gateway q3x::core
  )
  q3x_enable_warnings(q3x_openai_protocol_test)
  add_test(NAME openai_protocol COMMAND q3x_openai_protocol_test)

  add_executable(q3x_evaluation_server_cli_test
    tests/evaluation_server_cli_test.cpp
  )
  target_link_libraries(q3x_evaluation_server_cli_test
    PRIVATE q3x_eval_gateway
  )
  q3x_enable_warnings(q3x_evaluation_server_cli_test)
  add_test(NAME evaluation_server_cli
    COMMAND q3x_evaluation_server_cli_test
  )
  set_tests_properties(evaluation_server_cli PROPERTIES
    LABELS "host;server;profiling;compile_contract"
  )
  add_test(NAME eval_server_help COMMAND qwen3x-eval-server --help)

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

  add_executable(q3x_reference_prefill_single_arbitrary_tile_admission_e2e_test
    tests/reference_prefill_single_arbitrary_tile_admission_e2e_test.cpp
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

  add_executable(q3x_request_state_plan_test
    tests/request_state_plan_test.cpp
  )
  target_link_libraries(q3x_request_state_plan_test PRIVATE q3x::core)
  q3x_enable_warnings(q3x_request_state_plan_test)
  add_test(NAME request_state_plan COMMAND q3x_request_state_plan_test)

  add_executable(q3x_prefill_quantized_contract_test
    tests/prefill_quantized_contract_test.cpp
  )
  target_link_libraries(q3x_prefill_quantized_contract_test
    PRIVATE q3x::core
  )
  q3x_enable_warnings(q3x_prefill_quantized_contract_test)
  add_test(NAME prefill_quantized_contract
    COMMAND q3x_prefill_quantized_contract_test
  )
  set_tests_properties(prefill_quantized_contract PROPERTIES
    LABELS "host;prefill;architecture;manifest;arena"
  )

  add_executable(q3x_prefill_a4_sidecar_converter_test
    tests/prefill_a4_sidecar_converter_test.cpp
  )
  target_link_libraries(q3x_prefill_a4_sidecar_converter_test
    PRIVATE q3x::core
  )
  q3x_enable_warnings(q3x_prefill_a4_sidecar_converter_test)
  add_test(NAME prefill_a4_sidecar_converter
    COMMAND q3x_prefill_a4_sidecar_converter_test
  )
  set_tests_properties(prefill_a4_sidecar_converter PROPERTIES
    LABELS "host;prefill;quantization;sidecar;io"
  )

  if(Q3X_BUILD_LONG_PREFILL_LAYER_MAJOR_ADMISSION)
    add_executable(q3x_long_prefill_layer_major_plan_test
      tests/long_prefill_layer_major_plan_test.cpp
    )
    target_link_libraries(q3x_long_prefill_layer_major_plan_test
      PRIVATE q3x::core
    )
    q3x_enable_warnings(q3x_long_prefill_layer_major_plan_test)
    add_test(NAME long_prefill_layer_major_plan
      COMMAND q3x_long_prefill_layer_major_plan_test
    )
    set_tests_properties(long_prefill_layer_major_plan PROPERTIES
      LABELS "host;prefill;architecture;admission"
    )
  endif()

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
                qwen3x-orin qwen3x-eval-server qwen3x-inspect
                qwen3x-a4-sidecar
  EXPORT Qwen3xOrinTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
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
