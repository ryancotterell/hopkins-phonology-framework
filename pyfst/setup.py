import sys
import os
from distutils.core import setup
from distutils.extension import Extension

INC, LIB = [], []

# MacPorts
if sys.platform == 'darwin' and os.path.isdir('/opt/local/lib'):
    INC.append('/opt/local/include')
    LIB.append('/opt/local/lib')

ext_modules = [
    Extension(name='fst._fst',
              sources=['fst/_fst.cpp'],
              libraries=['fst'],
              extra_compile_args=['-O2','-std=c++11','-stdlib=libc++'],
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
