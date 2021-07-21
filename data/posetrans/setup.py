from os import name
from setuptools import setup
from Cython.Build import cythonize

setup(
    name="compute",
    ext_modules=cythonize('worldpos.pyx'),
    zip_safe=False
)
