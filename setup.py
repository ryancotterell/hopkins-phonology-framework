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
    Extension(name='pfst',
              sources=['templatic/pfst.cpp'],
              libraries=['fst', 'stdc++'],
              extra_compile_args=['-O2','-std=c++11','-stdlib=libc++', '-stdlib=libstdc++'],
              extra_link_args=['-lfst', '-lstdc++', '-L/mounts/Users/cisintern/ryan/openfst-1.4.1/src/lib/'],
              
              include_dirs=INC,
              library_dirs=LIB),
    Extension(name='variational_approximation',
              sources=['templatic/variational_approximation.cpp'],
              libraries=['fst', 'stdc++'],
              extra_compile_args=['-O2','-std=c++11','-stdlib=libc++', '-stdlib=libstdc++'],
              extra_link_args=['-lfst', '-lstdc++', '-L/mounts/Users/cisintern/ryan/openfst-1.4.1/src/lib/'],
              
        include_dirs=INC,
        library_dirs=LIB)
]


setup(
    name='templatic',
    author='Ryan Cotterell',
    packages=['templatic'],
    ext_modules=ext_modules
)
