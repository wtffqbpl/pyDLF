from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os
import subprocess
import shutil

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))
        
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={os.path.join(extdir, "pydlf")}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            f'-DPython3_ROOT_DIR={os.path.dirname(os.path.dirname(sys.executable))}',
            f'-DPython3_EXECUTABLE={sys.executable}',
            '-DCMAKE_BUILD_TYPE=Release'
        ]
        
        build_args = ['--', '-j4']
        
        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''),
            self.distribution.get_version())
        
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, 
                            cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args,
                            cwd=self.build_temp)
        
        module_path = os.path.join(self.build_temp, 'lib', 'pydlf', '_pydlf.cpython-39-darwin.so')
        if os.path.exists(module_path):
            shutil.copy2(module_path, os.path.join('pydlf', '_pydlf.cpython-39-darwin.so'))

setup(
    name='pydlf',
    version='0.1.0',
    author='Yuanjun Ren',
    author_email='renyuanjun310@gmail.com',
    description='A Python interface for the pyDLF C++ library',
    long_description='',
    ext_modules=[CMakeExtension('pydlf._pydlf')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    python_requires='>=3.9',
    packages=['pydlf'],
    package_dir={'pydlf': 'pydlf'},
) 