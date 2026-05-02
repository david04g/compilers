cmake_minimum_required(VERSION 3.20)

project(GpuSchedSim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(GPU_SCHED_SIM_SOURCES
  src/Config.cpp
  src/DependencyGraph.cpp
  src/Instruction.cpp
  src/Json.cpp
  src/Metrics.cpp
  src/Parser.cpp
  src/Scheduler.cpp
  src/Simulator.cpp
)

add_library(gpu_sched_sim_lib ${GPU_SCHED_SIM_SOURCES})
target_include_directories(gpu_sched_sim_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(gpu_sched_sim src/main.cpp)
target_link_libraries(gpu_sched_sim PRIVATE gpu_sched_sim_lib)

add_executable(gpu_sched_sim_tests
  tests/test_main.cpp
  tests/parser_tests.cpp
  tests/dependency_tests.cpp
  tests/scheduler_tests.cpp
  tests/simulator_tests.cpp
  tests/regression_tests.cpp
)
target_link_libraries(gpu_sched_sim_tests PRIVATE gpu_sched_sim_lib)

enable_testing()
add_test(NAME gpu_sched_sim_tests COMMAND gpu_sched_sim_tests)

add_custom_target(check-gpu-sched-sim
  COMMAND gpu_sched_sim_tests
  DEPENDS gpu_sched_sim_tests
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
