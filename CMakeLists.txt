cmake_minimum_required(VERSION 3.10)

project(pyDLF VERSION 0.1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug")
endif()

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Set compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

# Set position independent code for all targets
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find Python
set(Python_FIND_VIRTUALENV FIRST)
set(Python3_FIND_VIRTUALENV FIRST)
if(DEFINED ENV{VIRTUAL_ENV})
    set(Python3_ROOT_DIR $ENV{VIRTUAL_ENV})
endif()
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
message(STATUS "Python3_EXECUTABLE: ${Python3_EXECUTABLE}")
message(STATUS "Python3_INCLUDE_DIRS: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python3_LIBRARIES: ${Python3_LIBRARIES}")

# macOS specific settings
if(APPLE)
  execute_process(
    COMMAND xcrun --show-sdk-path
    OUTPUT_VARIABLE SYSROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  message(STATUS "macOS SDKROOT = ${SYSROOT}")

  set(CMAKE_OSX_SYSROOT "${SYSROOT}" CACHE PATH "macOS SDK root" FORCE)
  set(CMAKE_OSX_DEPLOYMENT_TARGET "12.0" CACHE STRING "macOS minimum deployment version" FORCE)

  # Set global compiler and linker flags for macOS
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isysroot ${SYSROOT} -stdlib=libc++ -nostdinc++")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -isysroot ${SYSROOT}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -stdlib=libc++")
  
  # Add C++ standard library include paths
  include_directories(SYSTEM
    "${SYSROOT}/usr/include/c++/v1"
    "${SYSROOT}/usr/include"
  )
endif()

# Find required packages
find_package(Armadillo REQUIRED)

# Math libraries
set(link_math_lib ${ARMADILLO_LIBRARIES})

set(dlf_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)

# Add project include directory
include_directories(${dlf_INCLUDE_DIR})

# Fetch dependencies
include(FetchContent)

# Fetch and configure spdlog
FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG v1.12.0
)

# Configure spdlog options
set(SPDLOG_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_PIC ON CACHE BOOL "" FORCE)
set(SPDLOG_NO_EXCEPTIONS ON CACHE BOOL "" FORCE)

if(APPLE)
  set(SPDLOG_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  set(SPDLOG_BUILD_SHARED OFF CACHE BOOL "" FORCE)
endif()

FetchContent_MakeAvailable(spdlog)

# Add spdlog include directory after FetchContent_MakeAvailable
include_directories(${spdlog_SOURCE_DIR}/include)

# Fetch and configure GoogleTest
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Fetch and configure pybind11
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.13.6
)
FetchContent_MakeAvailable(pybind11)

enable_testing()

# Set Python module name
set(PYTHON_MODULE_NAME "dlf")

# Add subdirectories
add_subdirectory(src)
add_subdirectory(tests)