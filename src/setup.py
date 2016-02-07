
## http://docs.cython.org/src/quickstart/build.html#building-a-cython-module-using-distutils

from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = 'KAGGLE-prudential-life-insurance-assessment',
  ext_modules = cythonize("kappa.pyx"),
)
