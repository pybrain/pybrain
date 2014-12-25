"""

    >>> from pybrain.tools.shortcuts import buildNetwork
    >>> from test_recurrent_network import buildRecurrentNetwork
    >>> from test_peephole_lstm import buildMinimalLSTMNetwork
    >>> from test_peephole_mdlstm import buildMinimalMDLSTMNetwork
    >>> from test_nested_network import buildNestedNetwork
    >>> from test_simple_lstm_network import buildSimpleLSTMNetwork
    >>> from test_simple_mdlstm import buildSimpleMDLSTMNetwork
    >>> from test_swiping_network import buildSwipingNetwork
    >>> from test_shared_connections import buildSharedCrossedNetwork
    >>> from test_sliced_connections import buildSlicedNetwork
    >>> from test_borderswipingnetwork import buildSimpleBorderSwipingNet

Test a number of network architectures, and compare if they produce the same output,
whether the Python implementation is used, or CTYPES.

Use the network construction scripts in other test files to build a number of networks,
and then test the equivalence of each.

Simple net
    >>> testEquivalence(buildNetwork(2,2))
    True

A lot of layers
    >>> net = buildNetwork(2,3,4,3,2,3,4,3,2)
    >>> testEquivalence(net)
    True

Nonstandard components
    >>> from pybrain.structure import TanhLayer
    >>> net = buildNetwork(2,3,2, bias = True, outclass = TanhLayer)
    >>> testEquivalence(net)
    True

Shared connections
    >>> net = buildSharedCrossedNetwork()
    >>> testEquivalence(net)
    True

Sliced connections
    >>> net = buildSlicedNetwork()
    >>> testEquivalence(net)
    True

Nested networks (not supposed to work yet!)
    >>> net = buildNestedNetwork()
    >>> testEquivalence(net)
    Network cannot be converted.

Recurrent networks
    >>> net = buildRecurrentNetwork()
    >>> net.name = '22'
    >>> net.params[:] = [1,1,0.5]
    >>> testEquivalence(net)
    True

Swiping networks
    >>> net = buildSwipingNetwork()
    >>> testEquivalence(net)
    True

Border-swiping networks
    >>> net = buildSimpleBorderSwipingNet()
    >>> testEquivalence(net)
    True

Lstm
    >>> net = buildSimpleLSTMNetwork()
    >>> testEquivalence(net)
    True

Mdlstm
    >>> net = buildSimpleMDLSTMNetwork()
    >>> testEquivalence(net)
    True

Lstm with peepholes
    >>> net = buildMinimalLSTMNetwork(True)
    >>> testEquivalence(net)
    True

Mdlstm with peepholes
    >>> net = buildMinimalMDLSTMNetwork(True)
    >>> testEquivalence(net)
    True


TODO:
- heavily nested
- exotic module use

"""

__author__ = 'Tom Schaul, tom@idsia.ch'
_dependencies = ['arac']

from pybrain.tests.helpers import buildAppropriateDataset, epsilonCheck
from pybrain.tests import runModuleTestSuite

def testEquivalence(net):
    cnet = net.convertToFastNetwork()
    if cnet == None:
        return None
    ds = buildAppropriateDataset(net)
    if net.sequential:
        for seq in ds:
            net.reset()
            cnet.reset()
            for input, _ in seq:
                res = net.activate(input)
                cres = cnet.activate(input)
                if net.name == '22':
                    h = net['hidden0']
                    ch = cnet['hidden0']
                    print(('ni', input, net.inputbuffer.T))
                    print(('ci', input, cnet.inputbuffer.T))
                    print(('hni', h.inputbuffer.T[0]))
                    print(('hci', ch.inputbuffer.T[0]))
                    print(('hnout', h.outputbuffer.T[0]))
                    print(('hcout', ch.outputbuffer.T[0]))
                    print()

    else:
        for input, _ in ds:
            res = net.activate(input)
            cres = cnet.activate(input)
    if epsilonCheck(sum(res - cres), 0.001):
        return True
    else:
        print(('in-net', net.inputbuffer.T))
        print(('in-arac', cnet.inputbuffer.T))
        print(('out-net', net.outputbuffer.T))
        print(('out-arac', cnet.outputbuffer.T))
        return (res, cres)


if __name__ == "__main__":
    runModuleTestSuite(__import__('__main__'))

