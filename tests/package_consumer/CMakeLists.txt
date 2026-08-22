cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(Q3xInstalledEngineConsumer LANGUAGES CXX)

include(CTest)
find_package(Qwen3xOrin 0.6.0 EXACT CONFIG REQUIRED)

add_executable(q3x_installed_engine_consumer main.cpp)
target_compile_features(q3x_installed_engine_consumer PRIVATE cxx_std_17)
target_link_libraries(q3x_installed_engine_consumer PRIVATE q3x::engine)

add_test(NAME installed_engine_consumer
  COMMAND q3x_installed_engine_consumer
)
