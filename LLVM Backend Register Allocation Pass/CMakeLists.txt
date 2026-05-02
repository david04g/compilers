cmake_minimum_required(VERSION 3.20)

project(CustomRegAlloc LANGUAGES C CXX)

find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT LLVM_INCLUDE_DIRS)
  get_filename_component(_CUSTOM_RA_LLVM_PREFIX "${LLVM_DIR}/../../.." ABSOLUTE)
  set(LLVM_INCLUDE_DIRS "${_CUSTOM_RA_LLVM_PREFIX}/include")
endif()

include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
separate_arguments(LLVM_DEFINITIONS_LIST UNIX_COMMAND "${LLVM_DEFINITIONS}")
add_definitions(${LLVM_DEFINITIONS_LIST})

add_llvm_library(CustomRegAlloc MODULE
  lib/Allocator.cpp
  lib/DebugEmitter.cpp
  lib/InterferenceGraph.cpp
  lib/Liveness.cpp
  lib/Pass.cpp

  PLUGIN_TOOL
  llc
)

target_include_directories(CustomRegAlloc
  SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS}
)
target_include_directories(CustomRegAlloc
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
)
if(TARGET obj.CustomRegAlloc)
  target_include_directories(obj.CustomRegAlloc
    SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS}
  )
  target_include_directories(obj.CustomRegAlloc
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
endif()

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

if(WIN32)
  set(CUSTOM_RA_PLUGIN
    "${CMAKE_CURRENT_BINARY_DIR}/CustomRegAlloc${CMAKE_SHARED_MODULE_SUFFIX}"
  )
else()
  set(CUSTOM_RA_PLUGIN
    "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_SHARED_MODULE_PREFIX}CustomRegAlloc${CMAKE_SHARED_MODULE_SUFFIX}"
  )
endif()
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
