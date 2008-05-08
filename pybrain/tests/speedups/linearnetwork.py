__author__ = 'Tom Schaul, tom@idsia.ch'

from scipy import randn
import time
import os


# execute the forward-pass a number of times and return the time needed
repeat = int(1500)
profiling = True
substitute = True
backward = False


if not substitute:
    os.environ["PYBRAIN_NO_SUBSTITUTE"] = 'True'



def buildLinearNetwork(hidden = 10, size = 10):
    """ build a network with some hidden linear layers """
    from pybrain import Network, LinearLayer, FullConnection
    N = Network()
    x = LinearLayer(size)
    N.addInputModule(x)
    for dummy in range(hidden+1):
        oldx = x
        x = LinearLayer(size)
        N.addModule(x)
        N.addConnection(FullConnection(oldx, x))
    N.sortModules()
    return N


N = buildLinearNetwork()
inp = randn(N.indim)
outerr = randn(N.outdim)

def main():
    for dummy in range(repeat):
        for dummy in range(3):
            N.activate(inp)
        if backward:
            for dummy in range(3):
                N.backActivate(outerr)
        N.reset()
        

if profiling:
    from pybrain.tests.helpers import sortedProfiling
    sortedProfiling('main()')    
else:
    start = time.time()
    main()
    print 'steps:', repeat, 'total time', time.time() - start
    