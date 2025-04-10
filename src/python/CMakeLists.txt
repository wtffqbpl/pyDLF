set(PYTHON_MODULE_NAME _pydlf)

pybind11_add_module(${PYTHON_MODULE_NAME} MODULE pydlf.cpp)
target_link_libraries(${PYTHON_MODULE_NAME} PRIVATE pydlf)
target_include_directories(${PYTHON_MODULE_NAME} PRIVATE ${Python3_INCLUDE_DIRS})

# Set output directory for the Python module
set_target_properties(${PYTHON_MODULE_NAME} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/pydlf"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/pydlf"
)

# Copy Python package files
file(COPY "${CMAKE_SOURCE_DIR}/pydlf" DESTINATION "${CMAKE_BINARY_DIR}"
     FILES_MATCHING PATTERN "*.py") 