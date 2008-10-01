#! /usr/bin/env python2.5
# -*- coding: utf-8 -*-


__author__ = 'Justin S Bayer, bayer.justin@googlemail.com'


import os
import sys

from setuptools import setup, find_packages
from distutils.ccompiler import new_compiler


class AracCompileError(Exception): pass


def compileArac():
    sources = [
        'arac/src/c/arac.c',
        'arac/src/c/common.c',
        'arac/src/c/functions.c',
        'arac/src/c/connections/common.c',
        'arac/src/c/connections/connections.c',
        'arac/src/c/connections/full.c',
        'arac/src/c/connections/identity.c',
        'arac/src/c/layers/bias.c',
        'arac/src/c/layers/common.c',
        'arac/src/c/layers/layers.c',
        'arac/src/c/layers/linear.c',
        'arac/src/c/layers/lstm.c',
        'arac/src/c/layers/mdlstm.c',
        'arac/src/c/layers/sigmoid.c',
        'arac/src/c/layers/softmax.c',
        'arac/src/c/layers/tanh.c',
        'arac/src/c/layers/mdrnn.c',
    ]
    
    compiler = new_compiler(verbose=1)

    if sys.platform.startswith('linux') or sys.platform == 'darwin':
        # Workaround for distutils to recognize .c files as c++files.
        compiler.language_map['.c'] = 'c++'
        compiler_cmd = 'g++'
        executables = {
            'preprocessor': None,
            'compiler': [compiler_cmd],
            'compiler_so': [compiler_cmd],
            'compiler_cxx': [compiler_cmd],
            'linker_so': [compiler_cmd, "-shared"],
            'linker_exe': [compiler_cmd],
            'archiver': ["ar", "-cr"],
            'ranlib': None,
        }
        compiler.set_executables(**executables)
        # Add some directories, this should maybe more sophisticated
        compiler.add_include_dir('/usr/local/include')
        compiler.add_include_dir('/usr/include')
        compiler.add_include_dir('/sw/include')
        compiler.add_include_dir('/sw/lib')
        compiler.add_library_dir('/usr/local/lib')
        compiler.add_library_dir('/usr/lib')
        output_dir = '/usr/local/lib'
    elif sys.platform.startswith('win'):
        raise AracCompileError("No support for arac on windows yet.")
    else:
        raise AracCompileError("Unknown platform: %s." % sys.platform)        
        
    compiler.add_library('m')
    compiler.add_library('blas')
    compiler.add_library('c')
    compiler.add_library('stdc++')
    objects = compiler.compile(sources, 
                               extra_postargs=['-O3', '-g0', '-DNDEBUG'])
    
    extra_postargs = ['-dynamiclib'] if sys.platform == 'darwin' else []
    extra_postargs += ['-O3']
    
    compiler.link_shared_lib(objects=objects, 
                             output_libname='arac', 
                             target_lang='c++', 
                             output_dir=output_dir,
                             extra_postargs=extra_postargs)


try:
    compileArac()
except AracCompileError, e:
    print "Fast networks are not available: %s" % e


setup(
    name="PyBrain",
    version="0.2pre",
    description="PyBrain is the swiss army knife for neural networking.",
    license="BSD",
    keywords="Neural Networks Machine Learning",
    url="http://pybrain.org",
    
    packages=find_packages(exclude=['examples', 'docs']) + 
             find_packages('./arac/src/python'),
    include_package_data=True,
    package_dir={'arac': './arac/src/python/arac/'},
    
    test_suite='pybrain.tests.runtests.make_test_suite',
)