__author__ = 'Tom Schaul, tom@idsia.ch'

from distutils.core import setup
from Pyrex.Distutils.extension import Extension
from Pyrex.Distutils import build_ext
import numpy
import sys
import os
from os.path import join
import platform

sys.argv.append('build_ext') 
sys.argv.append('--inplace') 

# Mac users need an environment variable set:
if platform.system() == 'Darwin':
    # get mac os version
    version = platform.mac_ver()
    # don't care about sub-versions like 10.5.2
    baseversion = version[0].rsplit('.',1)[0]
    os.environ["MACOSX_DEPLOYMENT_TARGET"] = baseversion

elif platform.system() == 'Windows':
    # It can be a bit tricky getting this to run...
    # Install Mingw and put MinGW\bin into the system path.
    # Tell it to Pyrex:
    sys.argv.append('-c') 
    sys.argv.append('mingw32') 
    # If this doesn't cut it, make sure (a copy of) e.g. 
    # python25.dll resides in Python25\libs\
    # not only in Windows\system32
    
    
structdir = join('..','structure')
files = {join(structdir,'modules'): ['_linearlayer', '_module', '_sigmoidlayer'],
         join(structdir,'connections'): ['_identity', '_connection', '_full'],
         join(structdir,'networks'): ['_network', ],
         '.': ['pyrex_shared'],
         }

# fetch all the files
extModules = []
libs = []
pyxFiles = []
for k, v in files.items():
    for fn in v:
        try:
            os.remove(join(k,fn+'.c'))
        except Exception: pass
        pyxfile = join(k,fn+'.pyx')
        pyxFiles.append(pyxfile)
        libs.append(fn)
        extModules.append(Extension(fn, [pyxfile],
                                    include_dirs = [numpy.get_include()],
                                    # all pxd files are here, in pybrain/tools/pyrex:
                                    pyrex_include_dirs = [os.path.abspath('.')],
                                    ))        
        
setup(ext_modules = extModules,
      cmdclass = {'build_ext': build_ext})


