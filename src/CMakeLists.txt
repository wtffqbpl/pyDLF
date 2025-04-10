set(target dlf)
add_library(${target} SHARED 
    dummy_code.cc
)

target_include_directories(${target}
        PRIVATE
        ${dlf_INCLUDE_DIR}
        ${spdlog_SOURCE_DIR}/include
        )

target_link_libraries(${target} PRIVATE 
        ${link_math_lib}
        spdlog::spdlog)
target_include_directories(${target} PRIVATE ${ARMADILLO_INCLUDE_DIR})

# Add the main library
add_library(pydlf SHARED
    tensor/tensor.cpp
    logger/logger.cpp
)

target_include_directories(pydlf PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(pydlf PUBLIC
    ${link_math_lib}
    spdlog::spdlog
)

# Add Python bindings
add_subdirectory(python)
