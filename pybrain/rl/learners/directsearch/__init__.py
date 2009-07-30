from rwr import RWR
from policygradients.__init__ import *

# also black-box optimizers
# TODO: this leads to circular imports...
try:
    from pybrain.optimization.__init__ import *
except:
    pass 