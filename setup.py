import setuptools  # triggers monkeypatching distutils
from distutils.core import setup
from os.path import dirname, join, abspath

import numpy as np
from Cython.Build import cythonize
from numpy.distutils.misc_util import get_info
from setuptools.extension import Extension

defs = [("NPY_NO_DEPRECATED_API", 0)]
np_inc_path = np.get_include()

extending = Extension(
    "pyddstore2",
    sources=["src/pyddstore.pyx", "src/ddstore.cxx"],
    include_dirs=[np_inc_path, "include", "/Users/jyc/sw/boost/1.80.0/include"],
    extra_compile_args=["-std=c++11"],
    define_macros=defs,
    library_dirs=[],
    libraries=["rt"],
)

extensions = [
    extending,
]

setup(
    name="PyDDStore2",
    version="0.1",
    description="Distributed Data Store",
    ext_modules=cythonize(extensions),
)
