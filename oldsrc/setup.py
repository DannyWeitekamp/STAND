# from setuptools import ... 
import sys,os
from os import path

from distutils.core import setup
# from Cython.Build import cythonize
from distutils.extension import Extension
import numpy
# import argparse



# C_SRC = "c_src"
# SRC = "src"
# LIB = "cython/lib"

# parser = argparse.ArgumentParser()
# parser.add_argument('--USE_CYTHON', dest='USE_CYTHON', action='store_const',
#                     help="compile the module using cython from .pyx source. If not specified then builds from .c files.", const=True, default=False)
# args = parser.parse_args()
USE_CYTHON = False
if "--USE_CYTHON" in sys.argv:
    USE_CYTHON = True
    sys.argv.remove("--USE_CYTHON")


if USE_CYTHON:
    from Cython.Build import cythonize
    import Cython.Compiler.Options
    Cython.Compiler.Options.annotate = True
    extensions = [Extension("*", [path.join("src","*.pyx")],extra_compile_args=["-w"])]
    extensions = cythonize(extensions, build_dir="c_src", annotate=True)
else:
	from glob import glob 
	sources = glob(path.join("c_src/src","*.c"))
	extensions = [Extension(path.splitext(path.basename(x))[0], [x]) for x in sources]

setup(
    ext_modules = extensions,
    include_dirs=[numpy.get_include()]
)