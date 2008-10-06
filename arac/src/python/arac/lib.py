"""Module that holds a handle to libarac."""

__all__ = ['libarac']

import os
import sys

import arac

libname = 'libarac.' + 'dll' if sys.platform.startswith('win') else 'so'
drct, _ = os.path.split(arac.__file__)
libpath = os.path.join(drct, libname)
libarac = ctypes.CDLL(libpath)
