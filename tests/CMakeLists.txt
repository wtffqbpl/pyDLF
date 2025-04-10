# Add C++ tests
add_subdirectory(csrctests)

# Add Python tests
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Configure Python test directory
set(PYTEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/pytests)

# Add custom target for Python tests
add_custom_target(pytest
    COMMAND ${Python3_EXECUTABLE} -m pytest ${PYTEST_DIR}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    DEPENDS _pydlf
    COMMENT "Running Python tests"
)

# Add Python tests to CTest
add_test(
    NAME pytest
    COMMAND ${Python3_EXECUTABLE} -m pytest ${PYTEST_DIR}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
) 