__author__ = 'Tom Schaul, tom@idsia.ch'

from distutils.core import setup
from Pyrex.Distutils.extension import Extension
from Pyrex.Distutils import build_ext
import numpy
import sys
import os
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
    sys.argv.append('-c') 
    sys.argv.append('mingw32') 
    

files = {'../structure/modules': ['_linearlayer', '_module', '_sigmoidlayer'],
         '../structure/connections': ['_identity', '_connection', '_full'],
         '../structure/networks': ['_network', ],
         '.': ['pyrex_shared'],
         }

# fetch all the files
extModules = []
libs = []
pyxFiles = []
for k, v in files.items():
    for fn in v:
        try:
            os.remove(k+'/'+fn+'.c')
        except Exception: pass
        pyxfile = k+'/'+fn+'.pyx'
        pyxFiles.append(pyxfile)
        libs.append(fn)
        extModules.append(Extension(fn, [pyxfile],
                                    include_dirs = [numpy.get_include()],
                                    # all pxd files are here, in pybrain/tools/pyrex:
                                    pyrex_include_dirs = [os.path.abspath('.')],
                                    ))        
        
setup(ext_modules = extModules,
      cmdclass = {'build_ext': build_ext})


