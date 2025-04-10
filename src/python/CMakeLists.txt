# Find Python and pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Set pybind11 path
set(pybind11_DIR "${Python3_SITEPACKAGES_DIR}/pybind11/share/cmake/pybind11")
find_package(pybind11 CONFIG REQUIRED)

# Add the Python module
pybind11_add_module(_dlf dlf.cpp)

# Link against the dlf library
target_link_libraries(_dlf PRIVATE dlf)

# Set the output directory
set_target_properties(_dlf PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Copy Python package files
file(COPY "${CMAKE_SOURCE_DIR}/pydlf" DESTINATION "${CMAKE_BINARY_DIR}"
     FILES_MATCHING PATTERN "*.py") 