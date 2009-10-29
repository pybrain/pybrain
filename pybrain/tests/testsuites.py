"""Module that contains several utilities for testing."""

__author__ = 'Justin Bayer, bayer.justin@googlemail.com'
__version__ = '$Id$'


from doctest import DocTestSuite, ELLIPSIS, REPORT_ONLY_FIRST_FAILURE, \
NORMALIZE_WHITESPACE
from unittest import TestSuite, TestLoader, TextTestRunner


def runModuleTestSuite(module):
    """Runs a test suite for all local tests."""
    suite = TestSuite([TestLoader().loadTestsFromModule(module)])

    # Add local doctests
    optionflags = ELLIPSIS | NORMALIZE_WHITESPACE | REPORT_ONLY_FIRST_FAILURE

    try:
        suite.addTest(DocTestSuite(module, optionflags=optionflags))
    except ValueError:
        # No tests have been found in that module.
        pass

    TextTestRunner().run(suite)

