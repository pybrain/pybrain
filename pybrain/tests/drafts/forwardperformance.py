import time
from scipy import ones

from complexnetwork import buildComplexNetwork
from pybrain.tools.shortcuts import buildSimpleNetwork


def testForwardPerformance(N, iterations = 1000):
    """ return ms per forward pass """
    start = time.time()
    r = ones(N.indim)
    for dummy in range(iterations):
        N.activate(r, 0)
    return (time.time() - start)*1000./iterations
    
    
if __name__ == '__main__':
    iterations = 5000
    print 'Tests over', iterations, 'iterations'
    print
    
    print 'simple:'
    
    N = buildSimpleNetwork(3, 3, 3)
    print 'params', N.paramdim,
    print 'modules', len(N.modules)
    print testForwardPerformance(N), 'ms'
    print
    
    N = buildSimpleNetwork(3, 1000, 3)
    print 'params', N.paramdim,
    print 'modules', len(N.modules)
    print testForwardPerformance(N), 'ms'
    print
    
    print 'complex:'
    
    N = buildComplexNetwork(300, 30)
    print 'params', N.paramdim,
    print 'modules', len(N.modules)
    print testForwardPerformance(N, 200), 'ms'
    print
    
    N = buildComplexNetwork(30, 300)
    print 'params', N.paramdim,
    print 'modules', len(N.modules)
    print testForwardPerformance(N, 500), 'ms'