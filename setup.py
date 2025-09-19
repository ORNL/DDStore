import setuptools  # triggers monkeypatching distutils
from distutils.core import setup
from os.path import dirname, join, abspath

import numpy as np
from Cython.Build import cythonize
from setuptools.extension import Extension

import os
import subprocess

defs = [('NPY_NO_DEPRECATED_API', 0)]
include_dirs = list()
library_dirs = list()
libraries = list()

## libfabric
libfabric_dir = subprocess.getoutput("pkg-config --variable=prefix libfabric")
libfabric_include_dir = os.path.join(libfabric_dir, "include")
if os.path.exists(os.path.join(libfabric_dir, "lib64")):
    libfabric_lib_dir = os.path.join(libfabric_dir, "lib64")
else:
    libfabric_lib_dir = os.path.join(libfabric_dir, "lib")
print("libfabric_dir:", libfabric_dir)
print("libfabric_include_dir:", libfabric_include_dir)
print("libfabric_lib_dir:", libfabric_lib_dir)
include_dirs.append(libfabric_include_dir)
library_dirs.append(libfabric_lib_dir)
libraries.append("fabric")

include_dirs.append(np.get_include())
include_dirs.append("include")

extending = Extension("pyddstore",
                      sources=["src/pyddstore.pyx", "src/ddstore.cxx", "src/common.cxx"],
                      include_dirs=include_dirs,
                      extra_compile_args=["-std=c++11"],
                      define_macros=defs,
                      library_dirs=library_dirs,
                      libraries=libraries,
                      )

extensions = [extending,]

setup(
    name="PyDDStore",
    version="0.1",
    description="Distributed Data Store",
    ext_modules=cythonize(extensions)
)
