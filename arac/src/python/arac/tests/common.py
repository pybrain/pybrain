#! /usr/bin/env python2.5
# -*- coding: utf-8 -*

"""Common functionality used by unittests."""


__author__ = 'Justin S Bayer, bayer.justin@googlemail.com'


import scipy
import unittest


class TestCase(unittest.TestCase):

    def assertArrayEqual(self, arr1, arr2):
        for a, b in zip(arr1, arr2):
            self.assertEqual(a, b)

    def assertArrayNear(self, arr1, arr2):
        self.assertEqual(scipy.size(arr1), scipy.size(arr2))
        for a, b in zip(arr1, arr2):
            self.assertNear(a, b)
            
    def assertNear(self, a, b, epsilon=0.0001):
        self.assert_(abs(a - b) < epsilon, 
                     "|%f - %f| > %f" % (a, b, epsilon))
        
        