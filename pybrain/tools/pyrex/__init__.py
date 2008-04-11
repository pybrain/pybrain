# Prefix with underscore to prevent export
import logging as _logging

try:
    from pyrex_shared import *
except ImportError, e:
    _logging.warning('Pyrex library seems to not exist. Try running the'
                     ' script pybrain/tools/pyrex/pyrexcompile.py')
