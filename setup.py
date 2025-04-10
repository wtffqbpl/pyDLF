import os
import sys
import subprocess

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import numpy as np

# Get the directory containing setup.py
setup_dir = os.path.abspath(os.path.dirname(__file__))

# Define the extension module
ext_modules = [
    Extension(
        "dlf._pydlf",  # Changed from dlf._dlf to match the import
        ["src/python/dlf.cpp"],  # Changed from pydlf.cpp
        include_dirs=[
            os.path.join(setup_dir, "include"),
            os.path.join(setup_dir, "include/tensor"),
            os.path.join(setup_dir, "include/utils"),
            np.get_include(),
        ],
        language="c++",
    )
]

# Custom build command to use CMake
class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name))
        )

        # Configure CMake
        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
            "-DPYTHON_EXECUTABLE=" + sys.executable,
            "-DCMAKE_BUILD_TYPE=Release",
        ]

        build_args = ["--", "-j4"]

        env = os.environ.copy()
        env["CXXFLAGS"] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get("CXXFLAGS", ""),
            self.distribution.get_version()
        )

        # Create build directory
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(
            ["cmake", setup_dir] + cmake_args,
            cwd=self.build_temp,
            env=env,
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args,
            cwd=self.build_temp,
        )

# Read the contents of README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="dlf",  # Changed from pydlf
    version="0.1.0",
    author="Yuanjun Ren",
    author_email="yuanjun.ren@outlook.com",
    description="A Python package for deep learning framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yuanjunren/pyDLF",
    packages=["dlf"],  # Changed from pydlf
    package_dir={"dlf": "dlf"},  # Changed from pydlf
    ext_modules=ext_modules,
    cmdclass=dict(build_ext=CMakeBuild),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "setuptools>=42.0.0",
        "wheel>=0.37.0",
    ],
) 