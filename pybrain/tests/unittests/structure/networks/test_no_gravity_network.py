"""
The library should be able to handle networks without any weight:

    >>> n1= buildNonGravityNet(False)
    >>> n1.paramdim
    0

    >>> n1.activate([0.2,0.4])[0]
    1.289...
    >>> n1.activate([0.2,0.4])[0]
    1.289...

Now let's verify the recurrent one as well:

    >>> n2= buildNonGravityNet(True)
    >>> n2.paramdim
    0

    >>> n2.activate([0.2,0.4])[0]
    1.289...
    >>> n2.activate([0.2,0.4])[0]
    3.478...

"""

from pybrain.structure import RecurrentNetwork, FeedForwardNetwork, IdentityConnection, LinearLayer, SigmoidLayer
from pybrain.tests.testsuites import runModuleTestSuite

def buildNonGravityNet(recurrent = False):
    if recurrent:
        net = RecurrentNetwork()
    else:
        net = FeedForwardNetwork()
    l1 = LinearLayer(2)
    l2 = LinearLayer(3)
    s1 = SigmoidLayer(2)
    l3 = LinearLayer(1)
    net.addInputModule(l1)
    net.addModule(l2)
    net.addModule(s1)
    net.addOutputModule(l3)
    net.addConnection(IdentityConnection(l1, l2, outSliceFrom = 1))
    net.addConnection(IdentityConnection(l1, l2, outSliceTo = 2))
    net.addConnection(IdentityConnection(l2, l3, inSliceFrom = 2))
    net.addConnection(IdentityConnection(l2, l3, inSliceTo = 1))
    net.addConnection(IdentityConnection(l1, s1))
    net.addConnection(IdentityConnection(l2, s1, inSliceFrom = 1))
    net.addConnection(IdentityConnection(s1, l3, inSliceFrom = 1))
    if recurrent:
        net.addRecurrentConnection(IdentityConnection(s1, l1))
        net.addRecurrentConnection(IdentityConnection(l2, l2, inSliceFrom = 1, outSliceTo = 2))
    net.sortModules()
    return net

if __name__ == "__main__":
    runModuleTestSuite(__import__('__main__'))

