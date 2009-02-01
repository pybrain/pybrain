#! /usr/bin/env python2.5
# -*- coding: utf-8 -*

"""Unittest runner script for arac."""


__author__ = 'Justin S Bayer, bayer.justin@googlemail.com'


import os
import logging
import doctest

from unittest import TestLoader, TestSuite, TextTestRunner


def make_test_suite():
    # The directory where the tests reside relative to the directory of this 
    # file.
    test_path_list = list(os.path.split(__file__)[:-1]) + ['arac/tests/']
    testdir = os.path.join(*test_path_list)

    # All unittest modules have to start with 'test_' and have to be, of 
    # course, python files
    module_names = [f[:-3] for f in os.listdir(testdir) 
                    if f.startswith('test_') and f.endswith('.py')]
                    
    if not module_names:
        logging.info("No tests found in %s" % testdir)
        sys.exit()
        
    # "Magically" import the tests package and its test-modules that we've 
    # found
    test_package_path = 'arac.tests'
    test_package = __import__(test_package_path, fromlist=module_names)
    
    # Put the test modules in a list that can be passed to the testsuite
    modules = [getattr(test_package, n) for n in module_names]
    
    # Print out a list of tests that are found
    for m in modules:
        logging.info("Tests found: %s" % m.__name__)
    
    # Build up the testsuite     
    suite = TestSuite([TestLoader().loadTestsFromModule(m) for m in modules])
    
    # Add doctests from the unittest modules to the suite
    optionflags = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
    for mod in modules:
        suite.addTest(doctest.DocTestSuite(mod, optionflags=optionflags))
        
    return suite
    

if __name__ == "__main__":
    runner = TextTestRunner()
    runner.run(make_test_suite())