"""

Determine if the modules in a Network are always sorted in the same way, even if the connections
don't constrain a particular order.

Build a number of modules and connections, to be used for all constructions

    >>> from pybrain.structure.networks.network import Network
    >>> mods = buildSomeModules(10)
    >>> conns = buildSomeConnections(mods)

Construct a network, normally, and sort it
    >>> n = Network()
    >>> for m in mods:
    ...    n.addModule(m)
    ...
    >>> for c in conns:
    ...    n.addConnection(c)
    ...
    >>> n.sortModules()
    >>> ord = str(n.modulesSorted)

Is the order the same, if we sort it again?

    >>> n.sortModules()
    >>> ord2 = str(n.modulesSorted)
    >>> ord == ord2
    True

What if we construct it in a different order?

    >>> n = Network()
    >>> for m in reversed(mods):
    ...    n.addModule(m)
    ...
    >>> for c in reversed(conns):
    ...    n.addConnection(c)
    ...
    >>> n.sortModules()
    >>> ord3 = str(n.modulesSorted)
    >>> ord == ord3
    True

Is it the same ordering than our reference?
    >>> print(ord3)
    [<LinearLayer 'l0'>, <LinearLayer 'l2'>, <LinearLayer 'l3'>, <LinearLayer 'l5'>, <LinearLayer 'l6'>, <LinearLayer 'l7'>, <LinearLayer 'l8'>, <LinearLayer 'l9'>, <LinearLayer 'l1'>, <LinearLayer 'l4'>]

"""

__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain import LinearLayer, FullConnection
from pybrain.tests import runModuleTestSuite


def buildSomeModules(number = 4):
    res = []
    for i in range(number):
        res.append(LinearLayer(1, 'l'+str(i)))
    return res

def buildSomeConnections(modules):
    """ add a connection from every second to every third module """
    res = []
    for i in range(len(modules)//3-1):
        res.append(FullConnection(modules[i*2], modules[i*3+1]))
    return res


if __name__ == "__main__":
    runModuleTestSuite(__import__('__main__'))

