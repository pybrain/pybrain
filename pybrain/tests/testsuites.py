"""
Contains several utilities for testing.
"""

from doctest import DocTestSuite, ELLIPSIS, REPORT_ONLY_FIRST_FAILURE, \
NORMALIZE_WHITESPACE
from unittest import TestSuite, TestLoader, TextTestRunner


def epsilonCheck(x, epsilon=1E-6):
    """Checks that x is in (-epsilon, epsilon)."""
    epsilon = abs(epsilon)
    return -epsilon < x < epsilon


def runModuleTestSuite(module):
    """Runs a test suite for all local tests."""
    suite = TestSuite([TestLoader().loadTestsFromModule(module)])

    # Add local doctests
    optionflags = ELLIPSIS | NORMALIZE_WHITESPACE | REPORT_ONLY_FIRST_FAILURE
    suite.addTest(DocTestSuite(module, optionflags=optionflags))

    TextTestRunner().run(suite)