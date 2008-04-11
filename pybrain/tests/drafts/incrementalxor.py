__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.rl.evolvables import BoundTotalInformation
from pybrain.tools.shortcuts import buildNetwork
from pybrain.rl.learners.searchprocesses import HillClimber, IncrementalComplexitySearch

from pybrain.examples.datasets import XORDataSet


ds = XORDataSet()
evaluator = lambda m: -ds.evaluateModuleMSE(m)

initm = BoundTotalInformation(buildNetwork(ds.indim, 5, ds.outdim), maxComplexity = 25)

sp = HillClimber(initm, evaluator)

inc = IncrementalComplexitySearch(sp, maxPhases = 6, desiredFitness = -0.1)
print inc.optimize()