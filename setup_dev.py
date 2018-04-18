from setuptools import setup
from Cython.Distutils import build_ext
from distutils.extension import Extension
import numpy

# options for enabling cython profiling
from Cython.Compiler.Options import get_directive_defaults
directive_defaults = get_directive_defaults()
directive_defaults['linetrace'] = True
directive_defaults['binding'] = True

extensions = [
    Extension("pycmf.cmf_newton_solver", ["pycmf/cmf_newton_solver.pyx"],
              define_macros=[('CYTHON_TRACE', '1')])
]

setup(
    name='pycmf',
    version='1.0.0',
    packages=['pycmf'],
    include_dirs=[numpy.get_include()],
    ext_modules=extensions,
    cmdclass={'build_ext': build_ext},
    url='https://github.com/keitakurita/smn-ailab/PyCMF',
    license='MIT',
    author='keitakurita',
    author_email='keita.kurita@gmail.com',
    description='A library for collective matrix factorization in Python.',
    install_requires=['scipy', 'numpy', 'scikit-learn'],
    classifiers=[
        'Programming Language :: Python :: 3.6'
    ]
)
