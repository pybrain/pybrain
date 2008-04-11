__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.rl.evolvables import PrecisionBoundParameters
from pybrain.tools.shortcuts import buildNetwork
from pybrain.rl.learners.searchprocesses import HillClimber

from pybrain.examples.datasets import XORDataSet

ds = XORDataSet()
evaluator = lambda m: -ds.evaluateModuleMSE(m)

initm = PrecisionBoundParameters(buildNetwork(ds.indim, 2, ds.outdim), maxComplexity = 3)

sp = HillClimber(initm, evaluator)

sp.search(1000, verbose = True)
