from distutils.core import setup, Extension
import numpy.distutils

p5_ext = Extension( 'p5.ext',
                    libraries = [],
                    include_dirs = ['../include'] +
                    numpy.distutils.misc_util.get_numpy_include_dirs(),
                    sources = [ 'p5_ext.cpp', '../src/p5_matrixA.cpp' ],
                    extra_compile_args = [ '-g' ]
                     )

setup( name = 'P5',
        version = '1.0',
        description = 'P5 library C++ extensions',
        ext_modules = [ p5_ext ] )
