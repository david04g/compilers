cmake_minimum_required(VERSION 3.20)

project(CustomRegAlloc LANGUAGES CXX)

find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
add_definitions(${LLVM_DEFINITIONS})

add_llvm_library(CustomRegAlloc MODULE
  lib/Allocator.cpp
  lib/DebugEmitter.cpp
  lib/InterferenceGraph.cpp
  lib/Liveness.cpp
  lib/Pass.cpp

  PLUGIN_TOOL
  llc
)

set_target_properties(CustomRegAlloc PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

if(NOT LLVM_LINK_LLVM_DYLIB)
  llvm_map_components_to_libnames(LLVM_LIBS
    CodeGen
    Core
    MC
    Support
    Target
  )
  target_link_libraries(CustomRegAlloc PRIVATE ${LLVM_LIBS})
endif()

set(CUSTOM_RA_PLUGIN
  "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_SHARED_MODULE_PREFIX}CustomRegAlloc${CMAKE_SHARED_MODULE_SUFFIX}"
)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/test/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/test/lit.site.cfg.py
  @ONLY
)

if(NOT LLVM_EXTERNAL_LIT)
  find_program(LLVM_EXTERNAL_LIT NAMES llvm-lit lit)
endif()

if(LLVM_EXTERNAL_LIT)
  add_custom_target(check-custom-ra
    COMMAND ${LLVM_EXTERNAL_LIT} -sv ${CMAKE_CURRENT_BINARY_DIR}/test
    DEPENDS CustomRegAlloc
    USES_TERMINAL
  )
else()
  message(WARNING "llvm-lit not found; check-custom-ra target is disabled")
endif()
