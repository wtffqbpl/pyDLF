cmake_minimum_required(VERSION 3.16)

project(PyDLF VERSION 0.1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(APPLE)
  execute_process(
    COMMAND xcrun --show-sdk-path
    OUTPUT_VARIABLE SYSROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  message(STATUS "macOS SDKROOT = ${SYSROOT}")

  set(CMAKE_OSX_SYSROOT "${SYSROOT}" CACHE PATH "macOS SDK root" FORCE)
  set(CMAKE_OSX_DEPLOYMENT_TARGET "12.0" CACHE STRING "macOS minimum deployment version" FORCE)

  set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -isysroot ${SYSROOT}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isysroot ${SYSROOT}")
endif()

include(FetchContent)
FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.16.0
        GIT_SHALLOW TRUE
)

# For Windows: Prevent overriding the parent project's compiler/linker settings.
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

# Libraries
find_package(Armadillo REQUIRED)

# Math libraries
set(link_math_lib ${ARMADILLO_LIBRARIES})

set(dlf_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)

add_subdirectory(src)
add_subdirectory(tests)