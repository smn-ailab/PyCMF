from setuptools import setup
from Cython.Distutils import build_ext
from distutils.extension import Extension
import numpy

extensions = [
    Extension("cmf_newton_solver", ["pycmf/cmf_newton_solver.pyx"])
]

setup(                                                                        
    name='pycmf',
    version='1.0.0',
    packages=['pycmf'],
    include_dirs = [numpy.get_include()],
    ext_modules=extensions,
    cmdclass={'build_ext': build_ext},
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
