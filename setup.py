import os
import sys
import subprocess
import sysconfig
import pybind11
import glob

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

# Get the directory containing setup.py
setup_dir = os.path.abspath(os.path.dirname(__file__))

# Define the extension module
ext_modules = [
    Extension(
        "dlf._pydlf",
        ["src/python/dlf.cpp"],
        include_dirs=[
            os.path.join(setup_dir, "include"),
            os.path.join(setup_dir, "include/tensor"),
            os.path.join(setup_dir, "include/utils"),
            pybind11.get_include(),
        ],
        language="c++",
        extra_compile_args=['-std=c++17'],
    )
]

# Custom build command to use CMake
class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_call(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                             ", ".join(e.name for e in self.extensions))

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # Get Python site-packages directory
        python_site_packages = sysconfig.get_paths()['purelib']
        pybind11_path = os.path.join(python_site_packages, 'pybind11')
        
        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
            '-DPYTHON_EXECUTABLE=' + sys.executable,
            '-DPYTHON_SITE_PACKAGES=' + python_site_packages,
            '-Dpybind11_DIR=' + os.path.join(pybind11_path, 'share/cmake/pybind11'),
            '-DCMAKE_PREFIX_PATH=' + os.path.join(pybind11_path, 'share/cmake')
        ]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

        # Copy the built extension to the package directory
        if not os.path.exists(os.path.dirname(self.get_ext_fullpath(ext.name))):
            os.makedirs(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # Find the built .so file
        so_file = glob.glob(os.path.join(extdir, '_pydlf*.so'))[0]
        self.copy_file(so_file, self.get_ext_fullpath(ext.name))

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

# Read the contents of README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="dlf",
    version="0.1.0",
    author="Yuanjun Ren",
    author_email="yuanjun.ren@outlook.com",
    description="A Python package for deep learning framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yuanjunren/pyDLF",
    packages=find_packages(),
    package_dir={"dlf": "dlf"},
    ext_modules=[CMakeExtension("dlf._pydlf")],
    cmdclass=dict(build_ext=CMakeBuild),
    package_data={
        'dlf': ['*.so', '_pydlf*.so'],  # Include .so files in the package
    },
    include_package_data=True,
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
    setup_requires=[
        "numpy>=1.24.0",
        "setuptools>=42.0.0",
        "wheel>=0.37.0",
        "pybind11>=2.13.0",
    ],
    install_requires=[
        "numpy>=1.24.0",
        "setuptools>=42.0.0",
        "wheel>=0.37.0",
        "pybind11>=2.13.0",
    ],
) 