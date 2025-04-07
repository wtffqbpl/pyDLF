set(target dlf)
add_library(${target} SHARED dummy_code.cc)

target_include_directories(${target}
        PRIVATE
        ${dlf_INCLUDE_DIR}
        )

target_link_libraries(${target} PRIVATE ${link_math_lib})
target_include_directories(${target} PRIVATE ${ARMADILLO_INCLUDE_DIR})
