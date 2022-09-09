"""
    >>> from pybrain.tests.helpers import epsilonCheck
    >>> from pybrain.tools.functions import tanh
    >>> from pybrain.utilities import fListToString
    >>> from .test_peephole_lstm import predictOutcome
    >>> from scipy import arctanh
    >>> from random import random

Test the MDLSTMLayer behavior when using peepholes.

    >>> N = buildMinimalMDLSTMNetwork()
    >>> N.params[:] = [.3,.4,.5]

    >>> s1 = 0.4
    >>> s2 = 0.414
    >>> s3 = -0.305
    >>> big = 10000

Set the state to s1
    >>> predictOutcome(N, [big, big, arctanh(s1), -big], 0)
    True

Verify that the state is conserved
    >>> predictOutcome(N, [-big, big, big*random(), big], tanh(s1))
    True

Add s2 to the state
    >>> predictOutcome(N, [big, big, arctanh(s2), big], tanh(s1+s2))
    True

Verify the peephole connection to the forgetgate (weight = .4) by neutralizing its contibution
and therefore dividing the state value by 2
    >>> predictOutcome(N, [-big, -(s1+s2) * .4, big*random(), big], tanh((s1+s2)/2))
    True

Verify the peephole connection to the inputgate (weight = .3) by neutralizing its contibution
and therefore dividing the provided input by 2. Also clearing the old state.
    >>> predictOutcome(N, [-(s1+s2)/2 * .3, -big, arctanh(s3), big], tanh(s3/2))
    True

Verify the peephole connection to the outputgate (weight = .5) by neutralizing its contibution
and therefore dividing the provided output by 2. Also clearing the old state.
    >>> predictOutcome(N, [-big, big, big*random(), -s3/2 * .5], tanh(s3/2)/2)
    True

List all the states again, explicitly (buffer size is 8 by now).
    >>> fListToString(N['mdlstm'].outputbuffer[:,1], 2)
    '[0.4  , 0.4  , 0.81 , 0.41 , -0.15, -0.15, 0    , 0    ]'

"""

__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.tests import runModuleTestSuite
from pybrain.structure import LinearLayer, IdentityConnection, MDLSTMLayer, RecurrentNetwork


def buildMinimalMDLSTMNetwork():
    N = RecurrentNetwork('simpleMdLstmNet')
    i = LinearLayer(4, name = 'i')
    h = MDLSTMLayer(1, peepholes = True, name = 'mdlstm')
    o = LinearLayer(1, name = 'o')
    N.addInputModule(i)
    N.addModule(h)
    N.addOutputModule(o)
    N.addConnection(IdentityConnection(i, h, outSliceTo = 4))
    N.addRecurrentConnection(IdentityConnection(h, h, outSliceFrom = 4, inSliceFrom = 1))
    N.addConnection(IdentityConnection(h, o, inSliceTo = 1))
    N.sortModules()
    return N

if __name__ == "__main__":
    runModuleTestSuite(__import__('__main__'))
