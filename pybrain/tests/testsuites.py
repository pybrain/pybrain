"""
Contains several utilities for testing.
"""

__author__ = 'Justin Bayer, bayer.justin@googlemail.com'


from doctest import DocTestSuite, ELLIPSIS, REPORT_ONLY_FIRST_FAILURE, \
NORMALIZE_WHITESPACE
from unittest import TestSuite, TestLoader, TextTestRunner


def runModuleTestSuite(module):
    """Runs a test suite for all local tests."""
    suite = TestSuite([TestLoader().loadTestsFromModule(module)])

    # Add local doctests
    optionflags = ELLIPSIS | NORMALIZE_WHITESPACE | REPORT_ONLY_FIRST_FAILURE
    suite.addTest(DocTestSuite(module, optionflags=optionflags))

    TextTestRunner().run(suite)