import sys
import os
from distutils.core import setup
from distutils.extension import Extension

INC, LIB = [], []
os.environ["CC"] = "clang++"
os.environ["CXX"] = "clang++"
# MacPorts

INC.extend(['/opt/local/include', '/mounts/Users/cisintern/ryan/openfst-1.4.1/src/include'])
LIB.extend(['/opt/local/lib', '/mounts/Users/cisintern/ryan/openfst-1.4.1/src/lib', '/mounts/Users/cisintern/ryan/openfst-1.4.1/src/bin'])

ext_modules = [
    Extension(name='fst._fst',
              sources=['fst/_fst.cpp', 'fst/pfst.cpp'],
              libraries=['fst', 'stdc++'],
              extra_compile_args=['-O2','-std=c++11','-stdlib=libc++', '-stdlib=libstdc++'],
              extra_link_args=['-lfst', '-lstdc++', '-L/mounts/Users/cisintern/ryan/openfst-1.4.1/src/lib/'],
              
        include_dirs=INC,
        library_dirs=LIB)
]

long_description='An extension to PyFST. Original code by Victor Chahuneau (http://pyfst.github.io/api.html).'
setup(
    name='pyfst',
    author='Ryan Cotterell',
    description='An extension to PyFST. Original code by Victor Chahuneau (http://pyfst.github.io/api.html).',
    long_description=long_description,
    classifiers=['Topic :: Text Processing :: Linguistic',
                 'Programming Language :: Cython',
                 'Programming Language :: C++',
                 'Intended Audience :: Education',
                 'Intended Audience :: Science/Research'],
    packages=['fst'],
    package_data={'fst':['_fst.pxd','libfst.pxd','sym.pxd','util.pxd']},
    ext_modules=ext_modules
)
