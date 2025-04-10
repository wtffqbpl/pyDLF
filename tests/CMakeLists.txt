# Add C++ tests
add_subdirectory(csrctests)

# Add Python tests
# Use the same Python interpreter as found in the main CMakeLists.txt
if(NOT Python3_EXECUTABLE)
    message(FATAL_ERROR "Python3_EXECUTABLE not set. Make sure Python3 is found in the main CMakeLists.txt")
endif()

# Configure Python test directory
set(PYTEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/pytests)

# Add custom target for Python tests
add_custom_target(pytest
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_SOURCE_DIR} ${Python3_EXECUTABLE} -m pytest ${PYTEST_DIR}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    DEPENDS _dlf
    COMMENT "Running Python tests"
)

# Add Python tests to CTest
add_test(
    NAME pytest
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_SOURCE_DIR} ${Python3_EXECUTABLE} -m pytest ${PYTEST_DIR}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
) 