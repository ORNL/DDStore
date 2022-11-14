import setuptools  # triggers monkeypatching distutils
from distutils.core import setup
from os.path import dirname, join, abspath

import numpy as np
from Cython.Build import cythonize
from numpy.distutils.misc_util import get_info
from setuptools.extension import Extension

defs = [('NPY_NO_DEPRECATED_API', 0)]
inc_path = np.get_include()

extending = Extension("pyddstore",
                      sources=["src/pyddstore.pyx", "src/ddstore.cxx"],
                      include_dirs=[np.get_include(), "include", "/Users/jyc/sw/boost/1.80.0/include"],
                      extra_compile_args=["-std=c++11"],
                      define_macros=defs,
                      library_dirs=[],
                      libraries=[],
                      )

extensions = [extending,]

setup(
    ext_modules=cythonize(extensions)
)