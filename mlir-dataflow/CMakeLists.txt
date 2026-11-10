cmake_minimum_required(VERSION 3.20)
project(MLIRDataflow LANGUAGES C CXX)
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using LLVM ${LLVM_PACKAGE_VERSION} and MLIR from ${MLIR_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(AddLLVM)
include(AddMLIR)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include_directories(include ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})
add_executable(dataflow-opt tools/dataflow-opt.cpp lib/PlanPass.cpp lib/Planner.cpp)
llvm_update_compile_flags(dataflow-opt)
target_link_libraries(dataflow-opt PRIVATE
  MLIRArithDialect MLIRFuncDialect MLIRLinalgDialect MLIRMathDialect
  MLIRTensorDialect MLIRIR MLIRParser MLIRPass MLIRSupport MLIROptLib)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
enable_testing()
add_test(NAME dataflow-tests COMMAND ${Python3_EXECUTABLE}
  -m unittest discover -s tests -v)
set_tests_properties(dataflow-tests PROPERTIES
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python;DATAFLOW_OPT=$<TARGET_FILE:dataflow-opt>")
