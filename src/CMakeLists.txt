add_executable(hello_armadillo main.cc)

target_link_libraries(hello_armadillo
    PRIVATE
        ${link_math_lib}
)
target_include_directories(hello_armadillo
    PRIVATE
        ${ARMADILLO_INCLUDE_DIR}
)
