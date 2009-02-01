#! /usr/bin/env python2.6
# -*- coding: utf-8 -*-


__author__ = 'Justin S Bayer, bayer.justin@googlemail.com'


import distutils
import glob
import os
import sys

from distutils.ccompiler import new_compiler
from setuptools import setup, find_packages

import numpy.distutils


def make_compiler(compiler_cmd='g++'):
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

    compiler = new_compiler(verbose=1)
    compiler.set_executables(**executables)
    compiler.add_include_dir('/usr/local/include')
    compiler.add_include_dir('/usr/include')
    compiler.add_include_dir('/sw/include')
    compiler.add_include_dir(distutils.sysconfig.get_python_inc())
    compiler.add_include_dir('/sw/lib')
    compiler.add_library_dir('/usr/local/lib')
    compiler.add_library_dir('/usr/lib')
    compiler.add_library_dir('.')

    for i in numpy.distutils.misc_util.get_numpy_include_dirs():
        compiler.add_include_dir(i)
    
    output_dir = '.'
        
    compiler.add_library('m')
    compiler.add_library('blas')
    return compiler


def compile_arac():
    if not (sys.platform.startswith('linux') or sys.platform == 'darwin'):
        raise AracCompileError('No support for arac on platform %s yet.' 
                               % sys.platform)
    
    globs = ('src/cpp/*.cpp', 
             'src/cpp/common/*.cpp', 
             'src/cpp/structure/*.cpp',  
             'src/cpp/structure/connections/*.cpp',  
             'src/cpp/structure/modules/*.cpp',  
             'src/cpp/structure/networks/*.cpp',
             'src/cpp/structure/networks/mdrnns/*.cpp',
    )
    sources = sum((glob.glob(i) for i in globs), [])

    compiler = make_compiler()
    objects = compiler.compile(sources, extra_postargs=['-g', '-O3'])
    compiler.add_library_dir('.')
    compiler.link_shared_lib(
        objects=objects, 
        output_libname='arac', 
        target_lang='c++', 
        extra_postargs=['-dynamiclib'] if sys.platform == 'darwin' else [])
    

def compile_swig():
    compiler = make_compiler()
    compiler.add_library('arac')
    objects = compiler.compile(['src/swig/cppbridge_wrap.cpp'],
                               extra_postargs=['-Wno-long-double'])
    compiler.link_shared_lib(
        objects=objects,
        output_dir='src/python/arac/',
        output_libname='_cppbridge',
        target_lang='c++',
        extra_postargs=['-bundle', 
                        '-undefined suppress', 
                        '-flat_namespace',
                        '-Wno-long-double']
    )
    os.rename('src/python/arac/lib_cppbridge.so', 
              'src/python/arac/_cppbridge.so')

    
def compile_test():    # Now compile test.                         
    compiler = make_compiler()
    objects = compiler.compile(glob.glob('src/cpp/tests/*.cpp'), 
                               extra_postargs=['-g', '-O3'])

    compiler.add_library('gtest')
    compiler.add_library('arac')
    compiler.link_executable(objects, 'test-arac')
