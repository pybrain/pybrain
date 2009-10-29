#! /usr/bin/env python2.5
# -*- coding: utf-8 -*- 

"""Script to run the pybrain testsuite."""

__author__ = 'Justin Bayer, bayerj@in.tum.de'
__version__ = '$Id$'


import doctest
import logging
import os
import sys

from unittest import TestLoader, TestSuite, TextTestRunner


def setUpLogging():
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s %(message)s')


def testImport(module_name):
    """Tell wether a module can be imported.
    
    This function has a cache, so modules are only tested once on 
    importability.
    """
    try:
        return testImport.cache[module_name]
    except KeyError:
        try:
            __import__(module_name)
        except ImportError:
            result = False
        else:
            result = True
    testImport.cache[module_name] = result
    return result
testImport.cache = {}   # Import checks are expensive, so cache results
        

def missingDependencies(target_module):
    """Returns a list of dependencies of the module that the current 
    interpreter cannot import.
    
    This does not inspect the code, but instead check for a list of strings
    called _dependencies in the target_module. This list should contain module
    names that the module depends on."""
    dependencies = getattr(target_module, '_dependencies', [])
    return [i for i in dependencies if not testImport(i)]


def make_test_suite():
    # The directory where the tests reside relative to the directory of this 
    # file.
    test_path_list = list(os.path.split(__file__)[:-1]) + ['unittests/']
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
    test_package_path = 'pybrain.tests.unittests'
    test_package = __import__(test_package_path, fromlist=module_names)
    
    # Put the test modules in a list that can be passed to the testsuite
    modules = (getattr(test_package, n) for n in module_names)
    modules = [(m, missingDependencies(m)) for m in modules]
    untests = [(m, md) for (m, md) in modules if md]
    modules = [m for (m, md) in modules if not md]
    
    # Print out modules that are missing dependencies
    for module, miss_dep in untests:    # Mr Dep is not around, though
        logging.warning("Module %s is missing dependencies: %s" % (
                        module.__name__, ', '.join(miss_dep)))
                        
    # Print out a list of tests that are found
    for m in modules:
        logging.info("Tests found: %s" % m.__name__)
    
    # Build up the testsuite     
    suite = TestSuite([TestLoader().loadTestsFromModule(m) for m in modules])
    
    # Add doctests from the unittest modules to the suite
    optionflags = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
    for mod in modules:
        try:
            suite.addTest(doctest.DocTestSuite(mod, optionflags=optionflags))
        except ValueError:
            # No tests found.
            pass
        
    return suite
    

if __name__ == "__main__":
    setUpLogging()
    runner = TextTestRunner()
    runner.run(make_test_suite())

