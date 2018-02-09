from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy

extensions = [
    Extension("cmf_newton_solver", ["cmf_newton_solver.pyx"])
]

setup(                                                                        
    name='CMF',
    version='1.0.0',
    packages=['CMF'],
    include_dirs = [numpy.get_include()],
    ext_modules=cythonize(extensions),
    url='https://github.com/keitakurita/CMF_Python',
    license='MIT',
    author='keitakurita',
    author_email='keita.kurita@gmail.com',
    description='A library for collective matrix factorization in Python.',   
    install_requires=['scipy', 'numpy', 'scikit-learn'],
    classifiers=[
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6'
    ]
)