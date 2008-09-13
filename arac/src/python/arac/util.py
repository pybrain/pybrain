#! /usr/bin/env python2.5
# -*- coding: utf-8 -*


__author__ = 'Justin S Bayer, bayer.justin@googlemail.com'

import math


def is_power_of_two(n):
    """Tell wether n is a power of two."""
    log = math.log(n, 2)
    return int(log) - log == 0.0