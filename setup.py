import setuptools  # triggers monkeypatching distutils
from distutils.core import setup
from os.path import dirname, join, abspath

import numpy as np
from Cython.Build import cythonize
from setuptools.extension import Extension

import os
import subprocess

defs = [("NPY_NO_DEPRECATED_API", 0)]
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

extending = Extension(
    "pyddstore._core",
    sources=["src/pyddstore/_core.pyx", "src/ddstore.cxx", "src/common.cxx"],
    include_dirs=include_dirs,
    extra_compile_args=["-std=c++11"],
    define_macros=defs,
    library_dirs=library_dirs,
    libraries=libraries,
)

extensions = [
    extending,
]

# The generated _core.cpp depends on the NumPy headers it was generated
# against: one generated with NumPy 2 does not compile against NumPy 1.x
# headers (PyDataType_ELSIZE). An editable or in-place build keeps it in
# src/, shared by every environment that builds from this checkout, so
# regenerate it whenever the NumPy major version differs from last time.
numpy_major = np.__version__.split(".")[0]
stamp = join("src", "pyddstore", "_core.numpy-version")
try:
    with open(stamp) as f:
        regenerate = f.read().strip() != numpy_major
except OSError:
    regenerate = True
with open(stamp, "w") as f:
    f.write(numpy_major + "\n")

# Left over from the layout before pyddstore became a package.
for old in ("src/pyddstore.cpp",) + tuple(
    join("src", f) for f in os.listdir("src")
    if f.startswith("pyddstore.") and f.endswith(".so")
):
    if os.path.exists(old):
        print(f"warning: stale build output {old} from the old layout; remove it")

setup(
    name="PyDDStore",
    version="2.0",
    description="Distributed Data Store",
    package_dir={"": "src"},
    packages=["pyddstore"],
    py_modules=["cpu_nic_map"],
    ext_modules=cythonize(extensions, force=regenerate),
)
