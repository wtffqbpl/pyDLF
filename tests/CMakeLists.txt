file(GLOB_RECURSE pydlf_test_srcs *.cc)

add_executable(pydlf_test ${pydlf_test_srcs})

target_link_libraries(pydlf_test
    PRIVATE
        ${link_math_lib}
)
target_include_directories(pydlf_test
    PRIVATE
        ${ARMADILLO_INCLUDE_DIR}
)


target_link_libraries(pydlf_test PRIVATE GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(pydlf_test)