__author__ = 'Tom Schaul, tom@idsia.ch'

from scipy import randn, zeros
import profile, pstats
    
from pybrain.structure.networks.network import Network
from pybrain.datasets import SequentialDataSet, SupervisedDataSet
from pybrain.supervised import BackpropTrainer
from pybrain.tools.xml import NetworkWriter, NetworkReader


def epsilonCheck(x, epsilon=1e-6):
    """Checks that x is in (-epsilon, epsilon)."""
    epsilon = abs(epsilon)
    return - epsilon < x < epsilon


def buildAppropriateDataset(module):
    """ build a sequential dataset with 2 sequences of 3 samples, with arndom input and target values,
    but the appropriate dimensions to be used on the provided module. """
    if module.sequential:
        d = SequentialDataSet(module.indim, module.outdim)
        for dummy in range(2):
            d.newSequence()
            for dummy in range(3):
                d.addSample(randn(module.indim), randn(module.outdim))
    else:
        d = SupervisedDataSet(module.indim, module.outdim)
        for dummy in range(3):
            d.addSample(randn(module.indim), randn(module.outdim))
    return d


def gradientCheck(module, tolerance=0.0001, dataset=None):
    """ check the gradient of a module with a randomly generated dataset, 
    (and, in the case of a network, determine which modules contain incorrect derivatives). """
    if module.paramdim == 0:
        print 'Module has no parameters'
        return True
    if dataset:
        d = dataset
    else:
        d = buildAppropriateDataset(module)
    b = BackpropTrainer(module)
    res = b._checkGradient(d, True)
    # compute average precision on every parameter
    precision = zeros(module.paramdim)
    for seqres in res:
        for i, p in enumerate(seqres):
            if p[0] == 0 and p[1] == 0:
                precision[i] = 0
            else:
                precision[i] += abs((p[0] + p[1]) / (p[0] - p[1]))
    precision /= len(res)
    if max(precision) < tolerance:
        print 'Perfect gradient'
        return True
    else:
        print 'Incorrect gradient', precision
        if isinstance(module, Network):            
            index = 0
            for m in module._containerIterator():
                if max(precision[index:index + m.paramdim]) > tolerance:
                    print 'Incorrect module:', m, res[-1][index:index + m.paramdim]
                index += m.paramdim
        else:
            print res
        return False
    
    
def xmlInvariance(n, forwardpasses=1):
    """ try writing a network to an xml file, reading it, rewrite it, reread it, and compare
    if the result looks the same (compare string representation, and forward processing 
    of some random inputs) """
    import os.path
    f = 'temp/xmlInvarianceTest.xml'
    if os.path.split(os.path.abspath(os.path.curdir))[1] == 'unittests':        
        f = '../' + f
    NetworkWriter.writeToFile(n, f)
    tmpnet = NetworkReader.readFrom(f)
    NetworkWriter.writeToFile(tmpnet, f)
    endnet = NetworkReader.readFrom(f)
    if str(n) == str(endnet):
        print 'Same representation'
    else:
        print n
        print "-" * 80
        print endnet
        
    outN = zeros(n.outdim)
    outEnd = zeros(endnet.outdim)
    n.reset()
    endnet.reset()
    for dummy in range(forwardpasses):
        inp = randn(n.indim)
        outN += n.activate(inp)
        outEnd += endnet.activate(inp)
        
    if sum(map(abs, outN - outEnd)) < 1e-9:
        print 'Same function'
    else:
        print outN
        print outEnd

    if n.__class__ == endnet.__class__:
        print 'Same class'
    else:        
        print n.__class__
        print endnet.__class__


def sortedProfiling(code, maxfunctions=50):
    import os.path
    f = 'temp/profilingInfo.tmp'
    if os.path.split(os.path.abspath(os.path.curdir))[1] != 'tests':        
        f = '../' + f
    profile.run(code, f)
    p = pstats.Stats(f)
    p.sort_stats('time').print_stats(maxfunctions)

