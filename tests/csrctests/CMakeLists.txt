file(GLOB_RECURSE pydlf_test_srcs *.cc)

include(GoogleTest)

add_executable(pydlf_test ${pydlf_test_srcs})

target_link_libraries(pydlf_test
    PRIVATE
        ${link_math_lib}
)
target_include_directories(pydlf_test
    PRIVATE
        ${ARMADILLO_INCLUDE_DIR}
        ${dlf_INCLUDE_DIR}
        ${dlf_INCLUDE_DIR}/utils
        ${spdlog_SOURCE_DIR}/include
)

enable_testing()

target_link_libraries(pydlf_test PRIVATE
        dlf_core
        GTest::gtest_main
        spdlog::spdlog)

gtest_discover_tests(pydlf_test)