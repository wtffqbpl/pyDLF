# Find Python and pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Find pybind11 using multiple methods
if(DEFINED pybind11_DIR)
    message(STATUS "Using provided pybind11_DIR: ${pybind11_DIR}")
    find_package(pybind11 CONFIG REQUIRED)
else()
    # Try to find pybind11 in Python site-packages
    if(DEFINED PYTHON_SITE_PACKAGES)
        set(pybind11_DIR "${PYTHON_SITE_PACKAGES}/pybind11/share/cmake/pybind11")
        message(STATUS "Looking for pybind11 in Python site-packages: ${pybind11_DIR}")
        find_package(pybind11 CONFIG QUIET)
    endif()

    # If not found, try system-wide installation
    if(NOT pybind11_FOUND)
        message(STATUS "Looking for system-wide pybind11")
        find_package(pybind11 CONFIG REQUIRED)
    endif()
endif()

message(STATUS "pybind11 found: ${pybind11_FOUND}")
message(STATUS "pybind11 version: ${pybind11_VERSION}")
message(STATUS "pybind11 include dirs: ${pybind11_INCLUDE_DIRS}")

# Add the Python module
pybind11_add_module(_pydlf 
    dlf.cpp
    ${CMAKE_SOURCE_DIR}/src/tensor/tensor_view.cpp
)

# Link against the dlf library
target_link_libraries(_pydlf PRIVATE dlf)

# Set RPATH for the module
if(APPLE)
    set_target_properties(_pydlf PROPERTIES
        INSTALL_RPATH "@loader_path"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()

# Set the output directory to be the Python package directory
set_target_properties(_pydlf PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/dlf"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/dlf"
    SUFFIX ".cpython-${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR}-darwin.so"
    PREFIX ""
)

# Copy Python package files
file(COPY "${CMAKE_SOURCE_DIR}/dlf" DESTINATION "${CMAKE_BINARY_DIR}/lib"
     FILES_MATCHING PATTERN "*.py") 