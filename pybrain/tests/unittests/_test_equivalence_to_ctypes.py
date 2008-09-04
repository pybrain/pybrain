"""

    >>> from pybrain import buildNetwork
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
    >>> net = buildNetwork(2,2)
    >>> testEquivalence(net)
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
    
Lstm
    >>> net = buildSimpleLSTMNetwork()
    >>> testEquivalence(net)
    True
    
Mdlstm
    >>> net = buildSimpleMDLSTMNetwork()
    >>> testEquivalence(net)
    True
    
Lstm with peepholes
    >>> net = buildMinimalLSTMNetwork()
    >>> testEquivalence(net)
    True
    
Mdlstm with peepholes
    >>> net = buildMinimalMDLSTMNetwork()
    >>> testEquivalence(net)
    True
    
Nested networks
    >>> net = buildNestedNetwork()
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
    
TODO: 
- MDLSTMs 
- heavily nested
- exotic module use


"""

__author__ = 'Tom Schaul, tom@idsia.ch'
_dependencies = ['arac']


from pybrain.structure import FeedForwardNetwork, RecurrentNetwork
from pybrain.tests.helpers import buildAppropriateDataset, epsilonCheck
from pybrain.tests import runModuleTestSuite

try:
    from arac.pybrainbridge import _RecurrentNetwork, _FeedForwardNetwork #@UnresolvedImport
except:
    pass

def convertToCImplemetation(net):
    net = net.copy()
    try:
        if isinstance(net, FeedForwardNetwork):
            cnet = _FeedForwardNetwork()
        elif isinstance(net, RecurrentNetwork):
            cnet = _RecurrentNetwork()
    except:
        return net
        
    for m in net.inmodules:
        cnet.addInputModule(m)
    for m in net.outmodules:
        cnet.addOutputModule(m)
    for m in net.modules:
        cnet.addModule(m)
        
    for clist in net.connections.values():
        for c in clist:
            cnet.addConnection(c)        
    if isinstance(net, RecurrentNetwork):
        for c in net.recurrentConns:
            cnet.addRecurrentConnection(c)
            
    cnet.sortModules()
    
    return cnet


def testEquivalence(net):
    cnet = convertToCImplemetation(net)
    ds = buildAppropriateDataset(net)
    if net.sequential:
        for seq in ds:
            for input, _ in seq:
                res = net.activate(input)
                cres = cnet.activate(input)
    else:
        for input, _ in ds:
            res = net.activate(input)
            cres = cnet.activate(input)
    return epsilonCheck(sum(res-cres))
    
        
if __name__ == "__main__":
    runModuleTestSuite(__import__('__main__'))

