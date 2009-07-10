#! /usr/bin/env python2.5
# -*- coding: utf-8 -*-


__author__ = 'Justin S Bayer, bayer.justin@googlemail.com'


import os
import sys

from setuptools import setup, find_packages

try:
    sys.path.append('./arac/')
    import aracsetuphelpers as aracsetup
    aracsetup.compile_arac()
    aracsetup.compile_swig()
except Exception, e:
    print "Fast networks are not available: %s" % e

setup(
    name="PyBrain",
    version="0.3pre",
    description="PyBrain is the swiss army knife for neural networking.",
    license="BSD",
    keywords="Neural Networks Machine Learning",
    url="http://pybrain.org",
    
    packages=find_packages(exclude=['examples', 'docs']) + 
             find_packages('./arac/src/python'),
    include_package_data=True,
    package_dir={'arac': './arac/src/python/arac'},
    package_data={'arac': ['_cppbridge.so']},
    data_files=[(os.path.join(sys.prefix, 'lib'), ['libarac.so'])],
    test_suite='pybrain.tests.runtests.make_test_suite',
)
