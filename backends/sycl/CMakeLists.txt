cmake_minimum_required(VERSION 3.10)

project(paddle-custom-sycl CXX C)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

option(WITH_ONEDNN "use oneDNN with custom kernels" ON)
#option(WITH_KERNELS "compile with custom kernels" ON)
option(ON_INFER "compile with inference c++ lib" OFF)
option(WITH_TESTING "compile with unit testing" ON)

get_filename_component(USER_HOME_DIR $ENV{HOME} ABSOLUTE)
if(NOT DEFINED ONEAPI_PATH)
	set(ONEAPI_PATH "${USER_HOME_DIR}/repos/paddle-onednn-dev")
endif()

if(NOT EXISTS "${ONEAPI_PATH}")
  message(FATAL_ERROR "Can't find openapi directory ${ONEAPI_PATH}")
endif()

message(STATUS "OneAPI DIR : ${ONEAPI_PATH}")

if(NOT DEFINED ONEAPI_COMPILER_DIR)
  set(ONEAPI_COMPILER_DIR "llvm-sycl")
endif()

if(NOT EXISTS "${ONEAPI_PATH}/${ONEAPI_COMPILER_DIR}")
  message(
    FATAL_ERROR "Can't find directory ${ONEAPI_PATH}/${ONEAPI_COMPILER_DIR}")
endif()

find_package(PythonInterp 3.9 REQUIRED)
find_package(PythonLibs 3.9 REQUIRED)

find_package(
  Python
  COMPONENTS Interpreter
  REQUIRED)

set(DPCPP_COMPIER_PATH "${ONEAPI_PATH}/${ONEAPI_COMPILER_DIR}/build/bin/clang++")
set(ONEAPI_SYCL_INCLUDE
    "${ONEAPI_PATH}/${ONEAPI_COMPILER_DIR}/build/include/sycl/")

set(SYCL_INCLUDE
    "${ONEAPI_PATH}/${ONEAPI_COMPILER_DIR}/build/include/")

set(ONEAPI_SYCL_LIBDIR "${ONEAPI_PATH}/${ONEAPI_COMPILER_DIR}/build/lib/")
set(CMAKE_CXX_COMPILER "${DPCPP_COMPIER_PATH}")

set(ONEDNN_INC "${ONEAPI_PATH}/oneDNN/include")
set(ONEDNN_BUILD_INC "${ONEAPI_PATH}/oneDNN/build/include")
set(ONEDNN_LIBDIR "${ONEAPI_PATH}/oneDNN/build/src")
message(${ONEDNN_INC})

if(WITH_ONEDNN)
	include_directories( ${SYCL_INCLUDE}  ${ONEAPI_SYCL_INCLUDE} ${ONEDNN_INC} ${ONEDNN_BUILD_INC})
else()
  include_directories(${ONEAPI_SYCL_INCLUDE} $ENV{DNNLROOT}/include)
endif()

set(PLUGIN_NAME "paddle-custom-sycl")
set(PLUGIN_VERSION "0.0.1")

include(paddle)
message(${PADDLE_INC_DIR})
message(STATUS "ONEDNN_INC DIR : ${ONEDNN_INC}")

include_directories(${PADDLE_INC_DIR} ${CMAKE_SOURCE_DIR}
                    ${THIRD_PARTY_INC_DIR} ${CMAKE_SOURCE_DIR}/kernels)
link_directories(${PADDLE_LIB_DIR} ${ONEAPI_SYCL_LIBDIR} ${ONEDNN_LIBDIR})

# build onednn from source
#if(WITH_ONEDNN)
#  add_definitions(-DONEDNN)
#endif()

add_definitions(-std=c++17 -fsycl -fsycl-unnamed-lambda -Waddress-of-temporary)

# custom kernels
file(
  GLOB_RECURSE PLUGIN_SRCS
  RELATIVE ${CMAKE_SOURCE_DIR}
  kernels/*.cc)
message(STATUS "PLUGIN_SRCS : ${PLUGIN_SRCS}")
list(APPEND PLUGIN_SRCS runtime/runtime.cc)
#add_definitions(-DPADDLE_WITH_CUSTOM_DEVICE)
#add_definitions(-DPADDLE_WITH_CUSTOM_KERNEL)

# build shared library
add_library(${PLUGIN_NAME} SHARED ${PLUGIN_SRCS})
if(ON_INFER)
  target_link_directories(${PLUGIN_NAME} PRIVATE ${PADDLE_INFERENCE_LIB_DIR})
  target_link_libraries(${PLUGIN_NAME} PRIVATE paddle_inference)
else()
  target_link_libraries(${PLUGIN_NAME} PRIVATE ${PADDLE_CORE_LIB})
endif()




# link paddle shared library
#include(third_party)
#add_dependencies(${PLUGIN_NAME} third_party)
target_link_libraries(${PLUGIN_NAME} PRIVATE ${PADDLE_CORE_LIB} ${ONEAPI_SYCL_LIBDIR}/libsycl.so ${ONEDNN_LIBDIR}/libdnnl.so)


if("${CMAKE_CXX_COMPILER}" STREQUAL "${DPCPP_COMPIER_PATH}")
  add_definitions(-DUSEDPCPP)
  set_target_properties(${PLUGIN_NAME} PROPERTIES CXX_STANDARD 17)
endif()

# packing wheel package
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in
               ${CMAKE_CURRENT_BINARY_DIR}/setup.py)

add_custom_command(
  TARGET ${PLUGIN_NAME}
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_CURRENT_BINARY_DIR}/python/
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/python/
  COMMAND ${CMAKE_COMMAND} -E make_directory
          ${CMAKE_CURRENT_BINARY_DIR}/python/paddle_custom_device/
  COMMAND
    ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_BINARY_DIR}/lib${PLUGIN_NAME}.so
    ${CMAKE_CURRENT_BINARY_DIR}/python/paddle_custom_device/
  COMMENT "Creating plugin directories------>>>")

find_package(
  Python
  COMPONENTS Interpreter
  REQUIRED)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/python/.timestamp
  COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/setup.py bdist_wheel
  DEPENDS ${PLUGIN_NAME}
  COMMENT "Packing whl packages------>>>")

add_custom_target(python_package ALL
                  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/python/.timestamp)

if(WITH_TESTING)
  set(PYTHON_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../Paddle")
  enable_testing()
  add_subdirectory(tests)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/tests/.timestamp
    COMMAND cp -r ${CMAKE_SOURCE_DIR}/tests ${CMAKE_CURRENT_BINARY_DIR})
  add_custom_target(python_tests ALL
                    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/tests/.timestamp)
endif()
