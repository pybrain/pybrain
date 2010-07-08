__author__ = 'Tom Schaul, tom@idsia.ch'


from distutils.core import setup
from Pyrex.Distutils.extension import Extension
from Pyrex.Distutils import build_ext
import platform, sys, os

# Mac users need an environment variable set:
if platform.system() == 'Darwin':
    # get mac os version
    version = platform.mac_ver()
    # don't care about sub-versions like 10.5.2
    baseversion = version[0].rsplit('.', 1)[0]
    os.environ["MACOSX_DEPLOYMENT_TARGET"] = baseversion

sys.argv.append('build_ext')
sys.argv.append('--inplace')

if platform.system() == 'Windows':
    sys.argv.append('-c')
    sys.argv.append('mingw32')

setup(ext_modules=[Extension('cartpolewrap', ['cartpolewrap.pyx', 'cartpole.cpp'],
                                pyrex_cplus=[True],
                                )],
      cmdclass={'build_ext': build_ext})

