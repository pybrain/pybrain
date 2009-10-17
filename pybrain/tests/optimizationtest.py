""" This test script will test the set of optimization algorithms.

It tests
 - the conformity of interface 
 - the behavior on simple functions
 - the behavior on FitnessEvaluators
 - the behavior when optimizing a list or an array
 - the behavior when optimizing an Evolvable
 - the behavior when optimizing a ParameterContainer
 - consistency w.r.t. minimization/maximization
 - tolerance of problems that have a constant fitness
 - tolerance of problems that have adversarial (strictly decreasing) fitness
 - handling one-dimensional and high-dimensional spaces
 - reasonable results on the linear function 
"""

__author__ = 'Tom Schaul, tom@idsia.ch'

from inspect import isclass
from scipy import sum, array, ndarray, log10
from random import random

import pybrain.optimization.optimizer as bbo
import pybrain.optimization.populationbased.multiobjective as mobj

from pybrain.rl.environments.functions.unimodal import SphereFunction
from pybrain.structure.parametercontainer import ParameterContainer
from pybrain.structure.evolvables.evolvable import Evolvable
from pybrain.rl.environments.cartpole.balancetask import BalanceTask
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules.module import Module
    

# Tasks to be optimized:
# ----------------------

# simple function
sf = lambda x: -sum((x+1)**2)
# FunctionEnvironment class
fe = SphereFunction
# initialized FE
ife1 = fe(1)
ife2 = fe(2)
ife100 = fe(100)
# a Task object
task = BalanceTask()
task.N = 10
# for the simple evolvable class defined below
evoEval = lambda e: e.x


# starting points
# ----------------------
xlist1 = [2.]
xlist2 = [0.2, 10]
xlist100 = list(range(12,112))

xa1 = array(xlist1)
xa2 = array(xlist2)
xa100 = array(xlist100)

pc1 = ParameterContainer(1)
pc2 = ParameterContainer(2)
pc100 = ParameterContainer(100)
pc1._setParameters(xa1)
pc2._setParameters(xa2)
pc100._setParameters(xa100)

# for the task object, we need a module
nnet = buildNetwork(task.outdim, 2, task.indim)
    
# a mimimalistic Evolvable subclass that is not (like usual) a ParameterContainer
class SimpleEvo(Evolvable):
    def __init__(self, x): self.x = x    
    def mutate(self):      self.x += random() - 0.3    
    def copy(self):        return evo(self.x)
    def randomize(self):   self.x = 10*random() - 2   
    
evo1 = SimpleEvo(-3.) 


# the test functions
# ----------------------

def testInterface(algo):
    """ Tests whether the algorithm is properly implementing the correct Blackbox-optimization interface."""
    # without any arguments, initialization has to work 
    emptyalgo = algo()
    try:
        # but not learning
        emptyalgo.learn(0)
        return "Failed to throw missing evaluator error?"
    except AssertionError:
        pass
    
    emptyalgo.setEvaluator(sf, xa1)
    # not it can run
    emptyalgo.learn(0)
            
    # simple functions don't check for dimension mismatch
    algo(sf, xa1)
    algo(sf, xa100)
    
    # for these, either an initial point or a dimension parameter is required
    algo(sf, numParameters = 2)
    
    try:
        algo(sf)
        return "Failed to throw unknown dimension error"
    except ValueError:
        pass 
    
    # FitnessEvaluators do not require that
    algo(ife1)        
    
    # parameter containers can be used too
    algo(ife2, pc2)
            
    return True

        
def testContinuousInterface(algo):
    """ Test the specifics for the interface for ContinuousOptimizers """
    if not issubclass(algo, bbo.ContinuousOptimizer):
        return True
    # list starting points are internally converted to arrays
    x = algo(sf, xlist2)
    assert isinstance(x.bestEvaluable, ndarray), 'not converted to array'
    
    # check for dimension mismatch
    try:
        algo(ife1, xa2)
        return "Failed to throw dimension mismatch error"
    except ValueError:
        pass
    
    return True
    

def testMinMax(algo):
    """ Verify that the algorithm is doing the minimization/maximization consistently. """
    if (issubclass(algo, bbo.TopologyOptimizer)
        or algo == StochasticHillClimber):
        # TODO
        return True
    
    xa1[0] = 2
    evalx = sf(xa1)    
    
    amax = algo(sf, xa1, minimize = False)
    assert amax.minimize is False or amax.mustMinimize, 'Max: Attribute not set correctly.'+str(amax.minimize)+str(amax.mustMinimize)
    x, xv = amax.learn(1)
    assert sf(x) == xv, 'Evaluation does not fit: '+str((sf(x), xv))
    assert xv >= evalx, 'Evaluation did not increase: '+str(xv)+' (init: '+str(evalx)+')'
    
    xa1[0] = 2
    amin = algo(sf, xa1, minimize = True)
    assert amin.minimize is True or amin.mustMaximize, 'Min: Attribute not set correctly.'+str(amin.minimize)+str(amin.mustMaximize)
    x, xv = amin.learn(1)
    assert sf(x) == xv, 'Evaluation does not fit: '+str((sf(x), xv))
    assert xv <= evalx, 'Evaluation did not decrease: '+str(xv)+' (init: '+str(evalx)+')'
    assert ((amin.minimize is not amax.minimize) 
            or not (amin.wasOpposed is amax.wasOpposed)), 'Inconsistent flags.' 
        
    return True
    
    


def testOnModuleAndTask(algo):
    l = algo(task, nnet)
    assert isinstance(l._bestFound()[0], Module), 'Did not return a module.'
    return True



class evo(Evolvable):
    def __init__(self, x): self.x = x    
    def mutate(self):      self.x += random() - 0.3    
    def copy(self):        return evo(self.x)
    def randomize(self):   self.x = 10*random() - 2
    
evoEval = lambda e: e.x
evo1 = evo(-3.) 

def testOnEvolvable(algo):
    if issubclass(algo, bbo.ContinuousOptimizer):
        return True
    if issubclass(algo, bbo.TopologyOptimizer):
        try:
            algo(evoEval, evo1).learn(1)
            return "Topology optimizers should not accept arbitrary Evolvables"
        except AttributeError:
            return True
    else:
        algo(evoEval, evo1).learn(1)
        return True



# the main test procedure
# ------------------------

def testAll(tests, allalgos, tolerant = True):
    countgood = 0
    for i, algo in enumerate(sorted(allalgos)):
        print "%d, %s:" % (i+1, algo.__name__)
        print ' '*int(log10(i+1)+2),
        good = True
        messages = []
        for t in tests:
            try:    
                res = t(algo)
            except Exception, e:
                if not tolerant:
                    raise e
                res = e
                
            if res is True:
                print '.',
            else:
                good = False
                messages.append(res)
                print 'F',
        if good:
            countgood += 1
            print '--- OK.'
        else:
            print '--- NOT OK.'
            for m in messages:
                if m is not None:
                    print ' '*int(log10(i+1)+2), '->', m
    print 
    print 'Summary:', countgood, '/', len(allalgos), 'passed all tests.'



if __name__ == '__main__':    
    from pybrain.optimization import *  #@UnusedWildImport
    
    allalgos = filter(lambda c: (isclass(c) 
                                 and issubclass(c, bbo.BlackBoxOptimizer)
                                 and not issubclass(c, mobj.MultiObjectiveGA) 
                                 ), 
                      globals().values())
    
    print 'Optimization algorithms to be tested:', len(allalgos)
    print    
    print 'Note: this collection of tests may take quite some time.'
    print 
    
    tests = [testInterface, 
             testContinuousInterface,
             testOnModuleAndTask,
             testOnEvolvable,
             testMinMax,
             ]
    
    testAll(tests, allalgos, tolerant = True)
