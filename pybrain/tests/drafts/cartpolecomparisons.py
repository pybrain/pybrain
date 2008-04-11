""" A script for trying plenty of variations of networks/algorithms on the cart-pole balancing task """

# WARNING! Be careful with lon runs: I supect there to be a fast-munching memory leak!


__author__ = 'Tom Schaul, tom@idsia.ch'

import time, copy
from scipy import rand, randn
from random import shuffle

from pybrain.tools.pyrex.cartpole.cartpoleenv import CartPoleTask
from pybrain.tools.shortcuts import buildNetwork
from pybrain import FullConnection
from nesexperiments import pickleDumpDict
from pybrain.rl.environments.functions.episodicevaluators.episodicevaluator import EpisodicEvaluator
from pybrain.rl.learners.blackboxoptimizers import CMAES
from pybrain.rl.environments.functions.function import OppositeFunction
from pybrain.rl.learners.searchprocesses.es import ES
from pybrain.rl.evolvables.evolvable import Evolvable
from pybrain.rl.learners.searchprocesses import MemeticHillClimber, HillClimber
from pybrain.rl.evolvables import MaskedParameters
from pybrain.rl.learners.searchprocesses.memeticcmaclimber import MemeticCMAClimber


# task: non-markov double pole balancing for 100000 iterations - fixed starting conditions
thetask = CartPoleTask(numPoles = 2, markov = False)

# desired performance (cumulative reward of the task)
desiredValue = 50000

# maximal number of episodes
maxEvals = 30000

# initial weight range
weightRange = 1.

shuffling = True

# networks: normal MLP with recurrent connections on hidden layer
# 2x2 variations - with and without bias, 3 or 6 hidden neurons.
networks = []
for h in [3,6]:
    for b in [True, False]:
        net = buildNetwork(thetask.getOutDim(), h, thetask.getInDim(), bias = b)
        net.addRecurrentConnection(FullConnection(net['hidden0'], net['hidden0'], name = 'rec'))
        net.sortModules()
        net.name = 'net'+str(h)+str(b)
        networks.append(net)
        # Tino-hack:
        net.params *= weightRange
if shuffling:
    shuffle(networks)


# a set of 6 algorithms. The interface for each one is:
# take a task and a network as input, optimize the network weights
# it returns a list of all parmeter settings, and their corresponding fitness
class EvolvableNetwork(Evolvable):
    """ the module must be a network - implements mutations (gaussian noise)"""
    variance = 0.2
    
    def mutate(self):
        self.reset()
        self.module.params += randn(self.module.paramdim) * self.variance
    
    def reset(self):
        # kill the old buffers
        for x in self.module.modules:
            x._resetBuffers()
        self.module._resetBuffers()
        Evolvable.reset(self)
        
    def copy(self):  
        self.reset()
        cp = Evolvable.copy(self)
        return cp
                      

def randomWeights(t, n, desired, maxEvals, verbose = False):
    best = 0
    e = EpisodicEvaluator(n, t)
    all = []
    for i in range(maxEvals):
        x = -weightRange + 2* weightRange * rand(n.paramdim)
        fit = e.controlledExecute(x)
        all.append((x, fit))
        if fit > best:
            best = fit
            if best > desired:
                break
        if verbose:    
            print i, best
    return all

def standardCMA(t, n, desired, maxEvals, verbose = False):
    e = OppositeFunction(EpisodicEvaluator(n, t))
    e.desiredValue = -desired
    x = -weightRange + 2* weightRange *rand(n.paramdim)    
    c = CMAES(e, x0 = x, maxEvals = maxEvals, silent = not verbose)
    c.optimize()
    return zip(e.xlist, map(lambda x: -x, e.vallist))

def standardES(t, n, desired, maxEvals, verbose = False):
    ev = EvolvableNetwork(n)
    n.params[:] = -weightRange + 2* weightRange *rand(n.paramdim)    
    res = []
    def evali(m):
        f = thetask.evaluateModule(m)
        res.append((m.module.params.copy(), f))
        return f
    a = ES(ev, evali, desiredFitness = desired)
    a.noisy = False
    a.search(maxEvals/a.lambada-2, verbose = verbose)
    return res
        
def hillClimber(t, n, desired, maxEvals, verbose = False):
    ev = EvolvableNetwork(n)
    n.params[:] = -weightRange + 2* weightRange *rand(n.paramdim)    
    res = []
    def evali(m):
        f = thetask.evaluateModule(m)
        res.append((m.module.params.copy(), f))
        return f
    a = HillClimber(ev, evali, desiredFitness = desired)
    a.noisy = False
    a.search(maxEvals-1, verbose = verbose)
    return res
        
def memeticClimber(t, n, desired, maxEvals, verbose = False):
    ev = MaskedParameters(n)
    n.params[:] = -weightRange + 2* weightRange *rand(n.paramdim)    
    res = []
    def evali(m):
        f = thetask.evaluateModule(m)
        res.append((m.module.params.copy(), f))
        return f
    a = MemeticHillClimber(ev, evali, localSteps = 50, desiredFitness = desired)
    a.noisy = False    
    a.search(maxEvals-41, verbose = verbose)
    return res

def memeticCMA(t, n, desired, maxEvals, verbose = False):
    ev = MaskedParameters(n)
    n.params[:] = -weightRange + 2* weightRange *rand(n.paramdim)    
    res = []
    def evali(m):
        f = thetask.evaluateModule(m)
        res.append((m.module.params.copy(), f))
        return f
    theothertask = copy.deepcopy(thetask)
    theothertask.evaluateModule = evali
    a = MemeticCMAClimber(ev, theothertask, localSteps = 50, desiredFitness = desired)
    a.noisy = False    
    a.search(maxEvals/14, verbose = verbose)
    return res



        
algorithms = [randomWeights, hillClimber, memeticClimber, 
              standardES, memeticCMA, standardCMA]
if shuffling:
    shuffle(algorithms)

# run all algorithms on all networks
def runAll(repeat = 1):
    for r in range(repeat):
        for n in networks:
            for a in algorithms:
                try:
                    start = time.time()
                    res = a(thetask, n, desiredValue, maxEvals, False)
                    # storage
                    pickleDumpDict('../temp/cartpole/'+n.name+'-'+a.__name__+'--'+str(int(rand(1)*90000)+10000), res)
                    t = time.time() - start
                    print n.name, a.__name__, 'episodes', len(res), 'best', max(map(lambda x: x[1], res)), 'time', t
                except Exception, e:
                    print 'Something went wrong', e
                
    
    
if __name__ == '__main__':
    if True:
        runAll(100)
        
    else:
        a, n = algorithms[5], networks[1]
        res = a(thetask, n, desiredValue, maxEvals, True)
        pickleDumpDict('../temp/cartpole/'+n.name+'-'+a.__name__+'--'+str(int(rand(1)*90000)+10000), res)                    
        print n.name, a.__name__, 'episodes', len(res), 'best', max(map(lambda x: x[1], res))
        
        
        